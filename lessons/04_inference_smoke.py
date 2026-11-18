"""Verify MLX inference with a small model, not cybersecurity expertise."""

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use cached model files only.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--prompt",
        default=(
            "Explain the difference between a threat and a vulnerability in two short sentences."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/runs/inference-smoke.json"))
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if not mx.metal.is_available():
        raise RuntimeError("This smoke test requires an available Apple Metal GPU.")
    mx.set_default_device(mx.gpu)

    model_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "73e3e38d981303bc594367cd910ea6eb48349da8"
    print(f"Model: {model_id}; offline={args.offline}", flush=True)
    started = time.perf_counter()
    snapshot = snapshot_download(repo_id=model_id, revision=revision, local_files_only=args.offline)
    download_seconds = time.perf_counter() - started
    started = time.perf_counter()
    model, tokenizer = load(snapshot, trust_remote_code=False)
    load_seconds = time.perf_counter() - started
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    mx.random.seed(42)
    started = time.perf_counter()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        sampler=make_sampler(temp=0.7, top_p=0.8, top_k=20),
        verbose=True,
    )
    generation_seconds = time.perf_counter() - started
    if not response.strip():
        raise RuntimeError("The model returned an empty response.")

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": model_id,
        "revision": Path(snapshot).name,
        "snapshot": Path(snapshot).name,
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "mlx_version": version("mlx"),
        "mlx_lm_version": version("mlx-lm"),
        "device": str(mx.default_device()),
        "offline": args.offline,
        "enable_thinking": False,
        "seed": 42,
        "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
        "max_tokens": args.max_tokens,
        "download_or_cache_lookup_seconds": download_seconds,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "peak_mlx_memory_gb": mx.get_peak_memory() / 1e9,
        "prompt": args.prompt,
        "response": response,
        "scope": "Non-empty generation smoke test; not a quality or completion benchmark.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: generated non-empty text on Metal. Report: {args.output}")


if __name__ == "__main__":
    main()
