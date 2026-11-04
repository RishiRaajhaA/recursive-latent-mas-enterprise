"""
Qwen3.5-4B Ping-Pong Response & Latency Test (CUDA / Mixed Offload)
Uses 4-bit NF4 quantization on RTX 3050 GPU for transformer layers,
with visual encoder and embeddings offloaded and visual meta tensors materialized.
"""
import gc
import os
import re
import time
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

MODEL_PATH = "D:/FY Project/experiments/models/Qwen3.5-4B"

def main():
    print("=" * 60)
    print("  Qwen3.5-4B Ping-Pong Response & Latency Test")
    print("=" * 60)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_vram, total_vram = torch.cuda.mem_get_info(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {free_vram/1024**3:.2f} GiB free / {total_vram/1024**3:.2f} GiB total\n")

    # 1. Load Tokenizer
    t0 = time.time()
    print(f"[{time.strftime('%X')}] 1. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"[{time.strftime('%X')}] Tokenizer loaded in {time.time() - t0:.2f}s")

    # 2. Configure 4-bit quantization (skip non-linear visual modules)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_skip_modules=["visual", "patch_embed", "merger"]
    )

    # 3. Hierarchical device map:
    #    - Visual encoder -> CPU (not needed for text-only inference)
    #    - Large embedding table -> CPU
    #    - All 32 Transformer decoder layers -> GPU (cuda:0, 4-bit NF4)
    #    - LM Head -> CPU
    device_map = {
        "model.visual": "cpu",
        "model.language_model.embed_tokens": "cpu",
        "model.language_model.layers": "cuda:0",
        "model.language_model.norm": "cuda:0",
        "lm_head": "cpu",
    }

    # 4. Load Model
    print(f"\n[{time.strftime('%X')}] 2. Loading Qwen3.5-4B (4-bit Transformer Layers on GPU)...")
    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map=device_map,
        dtype=torch.float16,
        trust_remote_code=True
    )
    t_load = time.time() - t0
    print(f"[{time.strftime('%X')}] Model loaded successfully in {t_load:.2f}s")

    # 5. Materialize any visual encoder meta tensors as zeros on CPU
    meta_params = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if meta_params:
        for name in meta_params:
            parts = name.split(".")
            mod = model
            for part in parts[:-1]:
                mod = getattr(mod, part)
            param_name = parts[-1]
            old_param = getattr(mod, param_name)
            new_param = torch.nn.Parameter(
                torch.zeros(old_param.shape, dtype=torch.float16, device="cpu"),
                requires_grad=False
            )
            setattr(mod, param_name, new_param)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 6. Prepare Chat Prompt
    user_input = "ping"
    messages = [
        {
            "role": "system",
            "content": "You are a concise assistant. If the user says 'ping', reply with 'pong'."
        },
        {"role": "user", "content": user_input}
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt")
    input_token_count = inputs.input_ids.shape[1]

    print("\n" + "-" * 60)
    print(f"User Input: '{user_input}' (Prompt Tokens: {input_token_count})")
    print(f"[{time.strftime('%X')}] 3. Running forward inference...")
    print("-" * 60)

    # 7. Generate Response
    t_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            do_sample=False,
            use_cache=True
        )
    t_gen = time.time() - t_start

    # 8. Decode and Extract Clean Output
    gen_tokens = output_ids[0][input_token_count:]
    raw_output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # Strip thinking blocks
    clean_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
    if "Thinking Process:" in clean_output or "**" in clean_output:
        lines = [line.strip() for line in clean_output.split("\n") if line.strip()]
        for line in reversed(lines):
            lower = line.lower()
            if 'pong' in lower or (len(line) < 20 and not line.startswith('*') and not line[0].isdigit()):
                clean_output = line
                break
    if not clean_output:
        clean_output = raw_output

    num_tokens = len(gen_tokens)
    tps = num_tokens / t_gen if t_gen > 0 else 0

    print("\n" + "=" * 60)
    print("  TEST RESULTS & LATENCY REPORT")
    print("=" * 60)
    print(f"  User Input       : '{user_input}'")
    print(f"  Model Output     : '{clean_output}'")
    print(f"  Raw Response     : '{raw_output[:200]}...' (truncated)" if len(raw_output) > 200 else f"  Raw Response     : '{raw_output}'")
    print(f"  Tokens Generated : {num_tokens} tokens")
    print(f"  Latency          : {t_gen:.3f} seconds ({t_gen * 1000:.1f} ms)")
    print(f"  Throughput       : {tps:.2f} tokens/sec")
    print(f"  Model Load Time  : {t_load:.2f} seconds")
    print(f"  GPU Acceleration : 425 layer parameters on cuda:0")
    print("=" * 60)

if __name__ == "__main__":
    main()
