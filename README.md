# ThreatTree Lab

Learn how a language model works, run one locally, build evidence-grounded attack
trees, and measure whether fine-tuning actually improves them.

**Recommended first model:** original
[`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B), using
[`Qwen/Qwen3-4B-MLX-4bit`](https://huggingface.co/Qwen/Qwen3-4B-MLX-4bit).
Use MLX-LM on an Apple Silicon Mac. Start in non-thinking mode; compare thinking
mode before deciding whether its additional latency helps your task.

## Current status: local inference verified

This project currently contains the folder structure, this README, a dependency
manifest, a resolved `uv.lock`, and four draft inference/training configurations.
One standalone inference lesson is implemented. Application code, fundamentals
lessons, datasets, and automated test suites have **not** been implemented.
The CLI declared in `pyproject.toml` is a future entry point, not a working command.
The configuration files are starting recipes, not locally tested training runs.

A project-local `.venv` now contains Python 3.12.16 (arm64), the core dependencies,
MLX training dependencies, and development tools. Package imports, dependency
compatibility, a tiny Metal GPU gradient calculation, and 0.6B model inference
have passed. The test model is cached locally, and offline inference also passed.
The planned 4B baseline and LoRA training have not been tested. No MITRE corpora
have been downloaded. This is an educational project, not a validated
cybersecurity product.

## Start here

Open this folder:

```sh
cd threattree-lab
```

Activate the environment in each new shell when you want to use its Python:

```sh
source .venv/bin/activate
python --version
```

Use `deactivate` to leave it. Activation is optional when running a tool by its
explicit path, for example `.venv/bin/python`.

To recreate or restore the installed dependencies from the lockfile:

```sh
uv sync --frozen --extra mlx --no-install-project
```

`--no-install-project` skips the unfinished application package, not its
dependencies. Keep `--extra mlx`: plain `uv sync` can remove the optional training
packages. To run an installed command without triggering a sync, use
`uv run --no-sync python --version`.

Begin with the smoke test and learning path below. The training tools are
installed, but the project CLI, fundamentals exercises, and training data do not
exist yet. Do not start a draft training config until you have prepared and
checked its dataset.

### Run the verified small-model smoke test

```sh
.venv/bin/python lessons/04_inference_smoke.py
```

This uses
[`mlx-community/Qwen3-0.6B-4bit`](https://huggingface.co/mlx-community/Qwen3-0.6B-4bit),
pinned to revision `73e3e38d981303bc594367cd910ea6eb48349da8`, not the larger
4B training candidate. It downloads on the first run and reuses the Hugging Face
cache thereafter. The initial download transferred about 334 MB.

To require cached files without Hugging Face network requests:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python lessons/04_inference_smoke.py --offline \
  --output artifacts/runs/inference-smoke-offline.json
```

Customize the prompt without editing the script:

```sh
.venv/bin/python lessons/04_inference_smoke.py --offline \
  --prompt "Explain multi-factor authentication in one sentence." \
  --max-tokens 96 --output artifacts/runs/mfa-smoke.json
```

The script disables thinking, bounds output length, and saves the prompt,
response, exact model revision, package versions, timings, and peak MLX memory.
The report at the selected output path is replaced on each run; choose another
`--output` path to preserve a previous experiment.

Both initial and offline runs generated non-empty text on `Device(gpu, 0)`,
with about 0.61 GB peak MLX memory. These short smoke runs establish that loading,
tokenization, chat templating, and GPU generation work; they are not benchmarks
or evidence of attack-tree quality. A non-empty response can still be wrong,
ignore instructions, or be truncated by the token cap.

### Preparation and toolchain

- **Python 3.12, native arm64:** use an isolated project environment rather than
  modifying system Python. See [Python](https://www.python.org/downloads/macos/).
- **uv:** dependency management with the resolved `uv.lock`.
  See [installation](https://docs.astral.sh/uv/getting-started/installation/).
- **NumPy:** small exercises for parameters, loss, gradients, and language models.
- **MLX-LM:** local model inference and quantized LoRA training on Apple Silicon.
  See [MLX installation](https://ml-explore.github.io/mlx/build/html/install.html)
  and [MLX-LM](https://github.com/ml-explore/mlx-lm).
- **Pydantic and Typer:** planned output validation and a small Python CLI.
- **SQLite FTS5:** planned local evidence retrieval; no vector database required.
- **pytest and Ruff:** planned offline tests, linting, and formatting.
- **JupyterLab and Matplotlib:** optional notebook and plotting tools, not installed.

The manifest separates the small core from the optional `mlx` and `learning`
extras. The `mlx` extra is installed; the `learning` extra is not required for
fine-tuning. Installed versions include MLX 0.32.4, MLX-Metal 0.32.4, MLX-LM
0.32.2, Transformers 5.23.0, Datasets 5.2.0, and NumPy 2.5.5. `uv pip check`
reported all 50 installed packages compatible. Docker, CUDA, hosted APIs, and
paid services are not required.

Additional model downloads will be explicit actions. Public ungated models normally require
no API key; cached weights can later be used offline. Model weights belong in the
Hugging Face cache, not Git. Preserve the exact model revision and tokenizer
alongside every trained adapter.

## What are you learning?

| Term | Meaning |
| --- | --- |
| Parameter / weight | A learned number inside the network. |
| Token | A text unit the model consumes or predicts; not necessarily a word. |
| Inference | Generating output with existing weights. |
| Loss | A numerical measure of prediction error during training. |
| Gradient | How the loss changes when a parameter changes. |
| SFT | Supervised fine-tuning: learning from reviewed input/answer pairs. |
| LoRA | Training small adapter matrices while freezing the original weights. |
| Quantized LoRA | LoRA over a compressed frozen model to save memory. |
| RAG | Looking up source material and providing it as prompt context. |
| Validation set | Cases used to choose prompts, settings, and checkpoints. |
| Test set | Separate cases reserved for the final frozen experiment. |

Fine-tuning teaches **behavior**, such as your tree structure, explanation style,
and uncertainty handling. Retrieval supplies **facts**, including current MITRE
identifiers and descriptions. Neither automatically guarantees correct analysis.

## Learning path

Follow these stages in order. The exercises and deeper project guides will be
added during implementation; the linked external tutorials are available now.

1. **Understand a training step.** Learn parameters, forward passes, loss,
   gradients, and learning rate. Start with
   [3Blue1Brown's neural-network lessons](https://www.3blue1brown.com/topics/neural-networks)
   and [Karpathy's Zero to Hero](https://karpathy.ai/zero-to-hero.html), beginning
   with micrograd. Goal: explain why changing a weight can lower prediction error.
2. **Build a tiny language model.** Follow Karpathy's makemore lessons using a
   small public text sample. Learn next-token prediction, train/validation
   splits, sampling, and overfitting. This tiny exercise is training from scratch;
   training a useful cybersecurity foundation model from scratch is not the goal.
3. **Understand pretrained models.** Study tokens, attention, context windows,
   instruction tuning, and chat templates in chapters 1–4 of the
   [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1).
   Goal: distinguish inference, pretraining, and fine-tuning.
4. **Run a baseline locally.** Use MLX-LM with the original Qwen3-4B. Compare fast
   and thinking modes on the same ten fictional scenarios. Save outputs, latency,
   token counts, and memory observations before changing the model.
5. **Add trustworthy data and evaluation.** Prepare versioned MITRE snapshots,
   reviewed scenario/answer pairs, graph validation, and a human scoring rubric.
   Compare supplied evidence and retrieved evidence before training.
6. **Perform a small LoRA experiment.** Read the
   [Hugging Face smol course](https://huggingface.co/learn/smol-course/en/unit1/1)
   for SFT/chat-template concepts and the
   [MLX-LM LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
   for the Mac toolchain. Do not copy CUDA-specific setup instructions into MLX.
7. **Evaluate and improve.** Compare the original model and adapter on held-out
   scenarios, analyze failures, and change one factor per experiment. Use an 8B
   comparison only after the smaller model's workflow is reproducible.

You do not need to finish a deep-learning textbook before trying inference.
You do need to understand loss, overfitting, data leakage, and chat templates
before interpreting fine-tuning results.

## Folder map

Directories below are reserved for their described contents. Apart from the
README, manifest, lockfile, draft configs, and scaffold markers, the application
and data directories remain initially empty. `lessons/04_inference_smoke.py`
is runnable, and `artifacts/runs/` contains its reports. `.venv/` contains
installed tools and is ignored by Git.

```text
threattree-lab/
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── docs/                       # Learning, model, data, and evaluation guides
├── lessons/
│   └── 04_inference_smoke.py    # Verified small-model inference
├── configs/
│   ├── inference-fast.yaml
│   ├── inference-thinking.yaml
│   ├── train-smoke.yaml
│   └── train-qlora.yaml
├── src/
│   └── threattree_lab/          # Future Python package
├── examples/
│   ├── scenarios/              # Fictional systems to analyze
│   └── expected/               # Reviewed reference answers
├── data/
│   ├── manifests/              # Provenance, versions, checksums, notices
│   ├── raw/                    # Downloaded snapshots; ignored by Git
│   ├── processed/              # Normalized evidence and indexes; ignored
│   ├── curated/                # Reviewed task-shaped training examples
│   └── splits/                 # Generated train/validation files; ignored
├── evals/                      # Development/test cases and scoring rubric
├── tests/
│   └── fixtures/               # Small model-free regression fixtures
└── artifacts/
    ├── adapters/               # Saved LoRA weights
    ├── runs/                   # Configs, timing, memory, and loss logs
    └── reports/                # Evaluation results and diagrams
```

## Model choice

The original Qwen3-4B is the first **learning baseline**, not a claim of best
cybersecurity performance. It has broad technical knowledge but is not a
verified or up-to-date MITRE database.

- [Original Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B): both thinking and
  non-thinking modes. Best starting point for the planned comparison.
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B): larger comparison if 4B
  decomposition quality is inadequate; measure the latency tradeoff.
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507):
  non-thinking-only alternative; not interchangeable with the original model.
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B): newer challenger.
  Verify training compatibility with the chosen MLX version before adopting its
  hybrid architecture; inference support does not prove training support.
- [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B): smaller
  dual-mode alternative with transparent training documentation.
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct):
  older non-thinking comparison, not the default for this project.

Check the exact publisher and model card for any checkpoint labeled Qwen3.8.
A community 4B distillation is not automatically an official small Qwen3.8.
These are downloadable/open-weight options; inspect each license and disclosure
level rather than assuming all training data and recipes are fully open.

Non-thinking does not mean incapable of reasoning. Thinking mode spends more
tokens before the answer and may or may not improve attack-tree correctness.
Start fast; enable thinking for an explicitly measured comparison.

## Data and evaluation preparation

Use [Enterprise ATT&CK STIX 2.1](https://github.com/mitre-attack/attack-stix-data)
for adversary behaviors and mitigations,
[CAPEC](https://capec.mitre.org/data/downloads.html) for attack patterns and
prerequisites, and [CWE](https://cwe.mitre.org/data/downloads.html) for weaknesses.
Read [CAPEC vs ATT&CK](https://capec.mitre.org/about/attack_comparison.html)
to understand their different purposes. Use
[NIST SP 800-30 Rev. 1](https://csrc.nist.gov/pubs/sp/800/30/r1/final)
as risk-assessment guidance, not as automatic labels for a training set.

Before training:

1. Record source URLs, snapshot versions/commits, dates, checksums, and terms.
   Retain attribution and exclude revoked/deprecated records by default.
2. Write task-shaped examples: system description plus desired JSON tree,
   assumptions, evidence references, mitigations, and uncertainty.
   Raw MITRE descriptions alone do not teach scenario decomposition.
3. Review 50 pilot examples manually before expanding toward 500–1,000 examples.
   AI-drafted examples need review and must be labeled honestly.
4. Deduplicate and group related scenarios before an 80/20 train/validation
   split. Keep final test scenarios separate; do not leak paraphrases between
   splits or use final-test failures to select hyperparameters.
5. Score JSON validity, graph correctness, AND/OR logic, prerequisite fit,
   source support, mitigation relevance, and uncertainty. Valid IDs alone do not
   prove a citation supports a claim.
6. Measure warm and cold latency separately, along with generated tokens,
   truncation, memory pressure, and swap. A lower training loss alone is not proof
   of a better threat model.

The planned application flow is: system description → retrieve evidence → model
output → validate JSON/tree → compute the documented risk rubric → render a
diagram and report. Validate one root, no cycles, reachable nodes, valid edges,
unique IDs, and explicit AND/OR semantics before rendering.

## Initial training recipe

The existing draft configs propose a frozen 4-bit Qwen3-4B base, LoRA rank 8 on
q/v attention projections in the last eight layers, microbatch 1, a 2,048-token
total sequence limit, gradient checkpointing, and assistant-only loss.

- `configs/train-smoke.yaml`: 20 microsteps, accumulation 1, for save/reload and
  finite-loss checks once data and code exist.
- `configs/train-qlora.yaml`: 200 microsteps, accumulation 4, as an initial pilot.
  This is only 50 optimizer updates, not a production training prescription.

MLX adapter scale is framework-specific; do not blindly substitute a different
framework's alpha. Reject or restructure overlength examples rather than
truncating an attack tree mid-JSON. Match the tokenizer/chat-template mode used
for training and inference. A fast-mode-only adapter is not evidence that the
same model retains its original thinking-mode quality.

## Mac memory and speed

Your 32 GB is shared by the OS, applications, weights, KV cache, and training
activations. A 4B model at 4 bits needs roughly 2 GB for weights alone before
metadata and unquantized tensors; **that is not a training-memory estimate**.
Start with one model, microbatch 1, short examples, and gradient checkpointing.
Keep Activity Monitor open and stop if memory pressure or swap grows persistently.
Do not change privileged wired-memory settings just to make an experiment fit.

No general tokens/second or training-time guarantee is made. Measure warm
generation separately from model download/load. Measure time to the first
**final-answer** token as well as the first reasoning token.

## Scope and limits

- Input: fictional web/API/identity/network systems, assets, controls, assumptions.
- Output: JSON attack trees, source references, mitigations, and qualitative risk.
- Illustrations are deterministic diagrams, not generated images.
- ATT&CK tactics are not a required temporal attack sequence.
- AND means all children are required; it does not encode ordering.
- Risk uses an explicitly educational 1–5 likelihood × 1–5 impact rubric.
  It is not a NIST-prescribed formula, calibrated probability, or compliance claim.
- No live scanning, exploit execution, automatic business decisions, or ISO/SAE
  21434 conformity claims.

## Next implementation milestones

1. Add the small Python fundamentals exercises and extend the verified 0.6B
   inference workflow to the planned 4B baseline.
2. Add output schemas, graph validators, reviewed examples, rendering, and
   model-free tests.
3. Add pinned-source ingestion, local retrieval, dataset validation, and
   leakage-resistant splitting.
4. Establish a baseline evaluation before running the smoke/pilot configs.
5. Compare baseline vs adapter, then fast vs thinking and 4B vs 8B, using the same
   scenarios and documented resource budgets.

The inference script has syntax and Ruff checks, plus actual online and offline
GPU smoke runs. A pytest suite, application CLI checks, and LoRA training
verification remain future work. No Git repository, commit, or remote publishing
is created by this setup.
