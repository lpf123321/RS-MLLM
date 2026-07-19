import atexit
import base64
import json
import math
import os
import socket
import subprocess
import sys
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from PIL import Image

from evaluation.base.adapter import BaseModelAdapter


# ===== Compatibility patches for vLLM + transformers 5.x =====
def _apply_patches():
    """Apply monkey-patches for known vLLM/transformers compatibility issues."""
    # Patch 1: Add all_special_tokens_extended to Qwen2Tokenizer if missing
    try:
        from transformers import AddedToken, AutoTokenizer
        import transformers.models.qwen2.tokenization_qwen2 as qwen2_tok
        if not hasattr(qwen2_tok.Qwen2Tokenizer, 'all_special_tokens_extended'):
            original_init = qwen2_tok.Qwen2Tokenizer.__init__

            def _patched_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self.all_special_tokens_extended = tuple(
                    AddedToken(t, single_word=False, lstrip=False, rstrip=False)
                    for t in self.all_special_tokens
                )

            qwen2_tok.Qwen2Tokenizer.__init__ = _patched_init
    except Exception:
        pass


_apply_patches()
# ===== End compatibility patches =====

_adapter_instance = None


def _cleanup_vllm():
    global _adapter_instance
    if _adapter_instance is not None:
        _adapter_instance.close()


atexit.register(_cleanup_vllm)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 4096 * 4096
MAX_TOOL_TURNS = 3

GEOEYES_SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call the zoom-in tool below to examine image details. Only use it when you
need to see fine details that are not visible at the current resolution.
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"image_index":{"type":"integer","description":"The index of the image to crop (0 for the first image, 1 for the second image, etc.). For single image samples, use 0."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox_2d"]}}}
</tools>

# Required Answer Format
You MUST end with the final answer inside <answer></answer> tags.
If you use a tool, first use <tool_call></tool_call>, then <answer></answer>.
If no tool is needed, output: <answer>your answer here</answer>
Do NOT include any text after the closing </answer> tag."""

USER_PROMPT_V2 = """
Always end with <answer>answer</answer>. Do not add text after </answer>."""


def smart_resize(
    height: int, width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    def _round(n, f): return round(n / f) * f
    def _ceil(n, f): return math.ceil(n / f) * f
    def _floor(n, f): return math.floor(n / f) * f

    h_bar = max(factor, _round(height, factor))
    w_bar = max(factor, _round(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor(height / beta, factor)
        w_bar = _floor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil(height * beta, factor)
        w_bar = _ceil(width * beta, factor)
    return h_bar, w_bar


def encode_image_to_base64(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_pil_image_to_base64(pil_image: Image.Image) -> str:
    buf = BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class GeoEyesAdapter(BaseModelAdapter):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 1024,
        vllm_port: Optional[int] = None,
        vllm_url: Optional[str] = None,
        gpu_memory_utilization: float = 0.92,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        system_prompt: str = "",
    ):
        global _adapter_instance
        self.max_new_tokens = max_new_tokens
        self.model_path = model_path
        self.system_prompt = system_prompt
        self._process = None
        self._log_file = None

        if vllm_url:
            self.api_base = vllm_url.rstrip("/")
        else:
            self.port = vllm_port or find_free_port()
            self.api_base = f"http://127.0.0.1:{self.port}/v1"
            self._start_vllm_server(
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
            )

        self.client = OpenAI(api_key="EMPTY", base_url=self.api_base)
        models = self.client.models.list()
        if not models.data:
            raise RuntimeError("vLLM server returned no models")
        self.model_name = models.data[0].id
        global _adapter_instance
        _adapter_instance = self

    def _start_vllm_server(self, gpu_memory_utilization, tensor_parallel_size, max_model_len):
        # V0 engine avoids flashinfer JIT compilation failure with GCC 8.5
        env = os.environ.copy()
        env["VLLM_USE_V1"] = "0"
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--model", self.model_path,
            "--served-model-name", "geoeyes",
            "--limit-mm-per-prompt", "image=4",
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--max-model-len", str(max_model_len),
            "--dtype", "bfloat16",
            "--trust-remote-code",
            "--enforce-eager",
        ]
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        self._log_file = os.path.join(log_dir, f"vllm_server_{self.port}.log")
        with open(self._log_file, "w") as lf:
            self._process = subprocess.Popen(
                cmd, stdout=lf, stderr=subprocess.STDOUT, env=env
            )
        self._wait_for_server(timeout=300)

    def _wait_for_server(self, timeout=300):
        import urllib.request
        start = time.time()
        url = f"{self.api_base}/models"
        while time.time() - start < timeout:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
            if self._process and self._process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server failed (exit code {self._process.returncode}). "
                    f"See log {self._log_file} for details."
                )
            time.sleep(5)
        raise TimeoutError(f"vLLM server not ready within {timeout}s")

    def close(self):
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def __del__(self):
        self.close()

    def generate(self, images: List[str], prompt: str) -> str:
        messages = [{"role": "system", "content": GEOEYES_SYSTEM_PROMPT}]

        # Prepend dataset-specific instructions if set
        if self.system_prompt:
            prompt = self.system_prompt + "\n" + prompt

        user_content = []
        for img_path in images:
            b64 = encode_image_to_base64(img_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        user_content.append({"type": "text", "text": prompt + USER_PROMPT_V2})
        messages.append({"role": "user", "content": user_content})

        pil_images = [Image.open(p) for p in images]

        last_content = ""
        for turn in range(MAX_TOOL_TURNS):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=self.max_new_tokens,
            )
            content = response.choices[0].message.content or ""
            last_content = content

            if "<answer>" in content and "</answer>" in content:
                return content.split("<answer>")[1].split("</answer>")[0].strip()

            if "<tool_call>" in content and "</tool_call>" in content:
                tool_str = content.split("<tool_call>")[1].split("</tool_call>")[0].strip()
                try:
                    tool_call = json.loads(tool_str)
                except json.JSONDecodeError:
                    break
                args = tool_call.get("arguments", {})
                bbox = args.get("bbox_2d")
                img_idx = args.get("image_index", 0)
                if not bbox or len(bbox) != 4:
                    break

                target_img = pil_images[img_idx] if 0 <= img_idx < len(pil_images) else pil_images[0]
                left, top, right, bottom = bbox
                cropped = target_img.crop((left, top, right, bottom))
                new_w, new_h = smart_resize(
                    bottom - top, right - left, factor=IMAGE_FACTOR
                )
                cropped = cropped.resize((new_w, new_h), Image.BICUBIC)
                cropped_b64 = encode_pil_image_to_base64(cropped)

                messages.append({"role": "assistant", "content": content})
                tool_content = [
                    {"type": "text", "text": "<tool_response>"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cropped_b64}"}},
                    {"type": "text", "text": USER_PROMPT_V2},
                    {"type": "text", "text": "</tool_response>"},
                ]
                messages.append({"role": "user", "content": tool_content})
            else:
                return content.strip()

        return last_content.strip()

    def batch_generate(self, batch, batch_size=1):
        from tqdm import tqdm
        results = []
        for images, prompt in tqdm(batch, desc="GeoEyes", unit="sample"):
            results.append(self.generate(images, prompt))
        return results
