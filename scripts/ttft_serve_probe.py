"""TTFT serve 探针：vLLM 原生 server + /metrics 取 vllm:time_to_first_token_seconds。
生成 1024x1024 测试图像 + 30 个 chat completion 请求（含图像），统计官方直方图。
"""
import base64
import io
import json
import os
import statistics
import time
import urllib.request

import numpy as np

SERVER = "http://127.0.0.1:8001"
RESULT = os.environ.get("OUTJSON", "/tmp/ttft_serve.json")


def make_image_b64() -> str:
    # 分辨率可配（IMG_PX=512/1024/4096），遥感风格的色块纹理图
    px = int(os.environ.get("IMG_PX", "1024"))
    img = np.zeros((px, px, 3), dtype=np.uint8)
    x, y = np.meshgrid(np.arange(px), np.arange(px))
    img[:, :, 0] = (60 + 40 * np.sin(x / 80.0)) % 256  # 纹理块
    img[:, :, 1] = (90 + 30 * np.cos(y / 60.0)) % 256
    img[:, :, 2] = 120
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def chat(body: dict) -> tuple[float, str]:
    req = urllib.request.Request(
        SERVER + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    e2e = time.perf_counter() - t0
    return e2e, data.get("choices", [{}])[0].get("message", {}).get("content", "")[:60]


def main() -> None:
    b64 = make_image_b64()
    url = "data:image/jpeg;base64," + b64
    serve_model = os.environ.get("SERVER_MODEL", "test")
    body = {
        "model": serve_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "请用一句话描述这幅遥感图像。"},
            ],
        }],
        "max_tokens": 20,
        "temperature": 0,
    }
    e2es = []
    for i in range(30):
        e2e, txt = chat(body)
        e2es.append(e2e)
        if i % 10 == 0:
            print(f"req {i}: e2e={e2e:.4f}s")
    print("e2e mean=%.4f p50=%.4f p95=%.4f" % (
        statistics.mean(e2es),
        statistics.median(e2es),
        sorted(e2es)[int(len(e2es) * 0.95) - 1],
    ))
    print("first_request_e2e(冷启动) = %.4f s" % e2es[0])

    # 拉取 /metrics 官方 TTFT 直方图
    with urllib.request.urlopen(SERVER + "/metrics", timeout=60) as r:
        text = r.read().decode()
    lines = [l for l in text.splitlines() if "time_to_first_token" in l]
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump({"e2e": e2es, "first_request_e2e": e2es[0],
                   "metrics": lines}, f, ensure_ascii=False, indent=2)
    print("=== vllm:time_to_first_token_seconds ===")
    for l in lines:
        print(l[:180])
    print("saved:", RESULT)


if __name__ == "__main__":
    main()
