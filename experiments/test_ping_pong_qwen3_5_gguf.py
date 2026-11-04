"""
Qwen3.5-4B GGUF Ping-Pong Response & Latency Test (llama.cpp CUDA)
Loads 100% of Qwen3.5-4B Q4_K_M onto RTX 3050 GPU using llama.cpp CUDA acceleration.
"""
import os
import re
import sys
import time
import zipfile
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(BASE_DIR, "llama_bin")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_REPO = "unsloth/Qwen3.5-4B-GGUF"
MODEL_FILE = "Qwen3.5-4B-Q4_K_M.gguf"

LLAMA_BIN_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b10718/llama-b10718-bin-win-cuda-12.4-x64.zip"
CUDART_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b10718/cudart-llama-bin-win-cuda-12.4-x64.zip"

def download_file_curl(url, dest_path):
    """Download a file using Windows curl.exe with progress."""
    print(f"  Downloading {os.path.basename(dest_path)}...")
    cmd = ["curl.exe", "-L", "-C", "-", "-o", dest_path, url]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to download {url}")

def ensure_llama_cuda_binaries():
    """Ensure llama.cpp CUDA binaries are downloaded and extracted."""
    os.makedirs(BIN_DIR, exist_ok=True)
    cli_path = os.path.join(BIN_DIR, "llama-completion.exe")
    if not os.path.exists(cli_path):
        cli_path = os.path.join(BIN_DIR, "llama-cli.exe")
    cuda_dll = os.path.join(BIN_DIR, "ggml-cuda.dll")
    if os.path.exists(cli_path) and os.path.exists(cuda_dll):
        return cli_path

    print(f"[{time.strftime('%X')}] Setting up llama.cpp CUDA 12.4 runtime...")
    for url in [LLAMA_BIN_URL, CUDART_URL]:
        fname = url.split("/")[-1]
        zip_path = os.path.join(BIN_DIR, fname)
        if not os.path.exists(zip_path) or not zipfile.is_zipfile(zip_path):
            if os.path.exists(zip_path):
                os.remove(zip_path)
            download_file_curl(url, zip_path)
        print(f"  Extracting {fname}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(BIN_DIR)
        try:
            os.remove(zip_path)
        except OSError:
            pass

    return cli_path

def ensure_model():
    """Ensure Qwen3.5-4B Q4_K_M GGUF is available."""
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    if os.path.exists(model_path) and os.path.getsize(model_path) > 2.0 * 1024**3:
        return model_path

    print(f"\n[{time.strftime('%X')}] Downloading {MODEL_FILE} (~2.55 GB) directly via curl...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"
    download_file_curl(url, model_path)
    return model_path

def main():
    print("=" * 60)
    print("  Qwen3.5-4B GGUF Ping-Pong Response & Latency Test (CUDA)")
    print("=" * 60)

    # 1. Setup binaries and model
    t0 = time.time()
    cli_path = ensure_llama_cuda_binaries()
    model_path = ensure_model()
    t_prep = time.time() - t0
    print(f"[{time.strftime('%X')}] Model & CUDA Engine ready in {t_prep:.2f}s")
    print(f"Model Path: {model_path} ({os.path.getsize(model_path) / 1024**3:.2f} GB)")

    # 2. Build prompt with chat template (pre-close think block for instant answers)
    user_input = sys.argv[1] if len(sys.argv) > 1 else "ping"
    system_prompt = "You are a helpful, direct, and concise AI assistant."
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"

    print("\n" + "-" * 60)
    print(f"System Prompt : '{system_prompt}'")
    print(f"User Query    : '{user_input}'")
    print(f"[{time.strftime('%X')}] Running 100% GPU Inference (-ngl 999) via llama.cpp CUDA...")
    print("-" * 60)

    # 3. Execute llama-completion with full GPU layer offloading
    cmd = [
        cli_path,
        "-no-cnv",            # Non-interactive completion
        "-m", model_path,
        "-p", prompt,
        "-ngl", "999",        # Offload all layers to RTX 3050 GPU
        "-n", "512",          # Max tokens
        "--temp", "0.0",      # Greedy sampling
        "--no-display-prompt" # Only output generated tokens
    ]

    t_start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    t_total = time.time() - t_start

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    # Combine output sources (llama.cpp outputs generated tokens to stdout or stderr depending on stream)
    raw_combined = stdout if stdout else ""
    if not raw_combined or len(raw_combined) < 5:
        # Extract response from stderr text
        for block in stderr.splitlines():
            if not block.startswith("0.") and not "common_perf" in block and not "llama_" in block:
                raw_combined += block + "\n"

    raw_output = raw_combined.replace("[end of text]", "").replace("<|im_end|>", "").strip()

    # Parse clean output (strip reasoning blocks)
    clean_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
    if not clean_output:
        clean_output = raw_output

    # 5. Extract llama.cpp timing metrics from stderr
    eval_tps = None
    eval_tokens = 0
    prompt_tps = None
    load_time_s = None

    for line in stderr.splitlines():
        if "load time =" in line:
            m = re.search(r"load time\s*=\s*([\d\.]+)\s*ms", line)
            if m:
                load_time_s = float(m.group(1)) / 1000.0
        if "prompt eval time =" in line:
            m = re.search(r"([\d\.]+)\s+tokens per second", line)
            if m:
                prompt_tps = float(m.group(1))
        if "eval time =" in line and "runs" in line:
            m = re.search(r"([\d\.]+)\s*ms\s*/\s*(\d+)\s*runs.*?\(\s*[\d\.]+\s*ms per token,\s*([\d\.]+)\s*tokens per second\)", line)
            if m:
                eval_tokens = int(m.group(2))
                eval_tps = float(m.group(3))

    print("\n" + "=" * 60)
    print("  TEST RESULTS & LATENCY REPORT (GGUF CUDA)")
    print("=" * 60)
    print(f"  User Input         : '{user_input}'")
    print(f"  Model Output       : '{clean_output}'")
    print(f"  Raw Response       : '{raw_output[:200]}...' (truncated)" if len(raw_output) > 200 else f"  Raw Response       : '{raw_output}'")
    if eval_tokens:
        print(f"  Tokens Generated   : {eval_tokens} tokens")
    if eval_tps:
        print(f"  Generation Speed   : {eval_tps:.2f} tokens/sec")
    if prompt_tps:
        print(f"  Prompt Eval Speed  : {prompt_tps:.2f} tokens/sec")
    if load_time_s:
        print(f"  Model Load Time    : {load_time_s:.2f} seconds")
    print(f"  Total Test Time    : {t_total:.3f} seconds ({t_total * 1000:.1f} ms)")
    print(f"  GPU Acceleration   : 100% Offload (RTX 3050 CUDA)")
    print("=" * 60)

if __name__ == "__main__":
    main()
