# p1_curves.py — measure load–latency curves for Paper 1 on Modal.
# Run:   modal run p1_curves.py
# Fetch: modal volume get p1-curves / ./curves
#
# Verify flags against your pinned vLLM version first:
#   vllm serve --help        (degradation flags)
#   vllm bench serve --help  (benchmark flags, saved metrics)

import subprocess
import time
import urllib.request

import modal

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # small open model; fits one L4
GPU = "L4"
RATES = [0.5, 1, 2, 3, 4, 6, 8, 10, 12, 14]

# Each config = one "zone health". Healthy is the default server;
# degraded versions shrink real capacity so the curve bends earlier.
CONFIGS = {
    "healthy": [],
    "degraded_seqs": ["--max-num-seqs", "16"],              # fewer concurrent requests
    "degraded_mem": ["--gpu-memory-utilization", "0.5"],    # smaller KV cache
}

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
results = modal.Volume.from_name("p1-curves", create_if_missing=True)
app = modal.App("p1-load-latency")


def wait_healthy(timeout_s=900):
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
            return
        except Exception:
            time.sleep(5)
    raise RuntimeError("vLLM server did not become healthy")


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,  # load generator shares the container; give it CPU headroom
    timeout=3 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/results": results},
)
def sweep(config: str, repeat: int = 0):
    server = subprocess.Popen(
        ["vllm", "serve", MODEL, "--port", "8000", "--max-model-len", "8192",
         *CONFIGS[config]]
    )
    wait_healthy()
    for rate in RATES:
        subprocess.run(
            [
                "vllm", "bench", "serve",
                "--base-url", "http://127.0.0.1:8000",
                "--model", MODEL,
                "--dataset-name", "random",
                # Copilot-like shape, scaled down: long shared prefix + short output
                "--random-prefix-len", "3000",
                "--random-input-len", "1000",
                "--random-output-len", "250",
                "--request-rate", str(rate),
                "--num-prompts", str(max(50, int(rate * 120))),  # change it back to 2 minutes 
                "--save-result",
                "--result-dir", "/results",
                "--result-filename", f"{config}_rep{repeat}_rate{rate}.json",
                
            ],
            check=True,
        )
    server.terminate()
    results.commit()


@app.local_entrypoint()
def main():
    # Runs the three configs in parallel, one GPU each.
    list(sweep.map(list(CONFIGS.keys())))