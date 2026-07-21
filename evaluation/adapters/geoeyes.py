import atexit
import base64
import os
import socket
import subprocess
import sys
import time
from io import BytesIO
from typing import List, Optional, Tuple

from openai import OpenAI
from PIL import Image

from evaluation.base.adapter import BaseModelAdapter


# ===== Compatibility patches for vLLM + transformers 5.x =====
def _apply_patches():
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


def encode_image_to_base64(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG")
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
        max_new_tokens: int = 256,
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
        _adapter_instance = self

    def _start_vllm_server(self, gpu_memory_utilization, tensor_parallel_size, max_model_len):
        env = os.environ.copy()
        env["VLLM_USE_V1"] = "0"
        # Ensure sitecustomize.py is loaded for transformers compat patches
        patches_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_patches")
        env["PYTHONPATH"] = patches_dir + os.pathsep + env.get("PYTHONPATH", "")
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
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        user_content = []
        for img_path in images:
            b64 = encode_image_to_base64(img_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=self.max_new_tokens,
        )
        content = response.choices[0].message.content or ""

        if "<answer>" in content and "</answer>" in content:
            return content.split("<answer>")[1].split("</answer>")[0].strip()
        return content.strip()

    def batch_generate(self, batch: List[Tuple[List[str], str]], batch_size: int = 1) -> List[str]:
        from tqdm import tqdm
        results = []
        for images, prompt in tqdm(batch, desc="GeoEyes", unit="sample"):
            results.append(self.generate(images, prompt))
        return results
