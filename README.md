# LettuceDetect Reimplementation

Sem 5 minor research project: a from-scratch reimplementation of
[LettuceDetect](https://arxiv.org/abs/2502.17125) (Kovacs & Recski, 2025), a
token-level hallucination detector for RAG systems built on ModernBERT, plus
an original extension studying cross-dataset generalization against
[PsiloQA](https://arxiv.org/abs/2510.04849).

See [`docs/project_brief.md`](docs/project_brief.md) for the full project
scope and phase plan, and
[`docs/literature_survey_and_presentation.md`](docs/literature_survey_and_presentation.md)
for the literature survey and presentation material.

## Workflow: code here, train on the RTX 4060 machine

This repo is developed on a machine without a dedicated GPU. The convention:

- **All code, data-prep logic, and small CPU-only smoke tests** are written
  and committed from any machine.
- **Real training runs** (full RAGTruth / PsiloQA fine-tuning) happen only on
  the RTX 4060 laptop, which clones this repo and pulls the latest commits
  before each run.
- **Never commit:** raw or processed datasets, model checkpoints, or anything
  under `data/`, `checkpoints/`, or `results/*.json` beyond the placeholders
  already tracked. See `.gitignore`.
- Every script resolves paths relative to the repo root at runtime (via
  `pathlib.Path(__file__).resolve()`), not the current working directory or a
  hardcoded absolute path - so the same code runs correctly on both machines
  without editing paths by hand.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` does **not** pin a specific PyTorch build, because the
correct build differs by machine (CPU-only here, CUDA-enabled on the RTX 4060
laptop). Install PyTorch separately first, matching your hardware, following
the selector at https://pytorch.org/get-started/locally/ - then run the
`pip install -r requirements.txt` above for everything else.

## Repo layout

```
lettucedetect-reimpl/
├── docs/                 # brief, literature survey, original paper PDF
├── data/
│   ├── raw/              # downloaded RAGTruth / PsiloQA JSON (gitignored)
│   └── processed/        # tokenized + label-aligned cache (gitignored)
├── src/
│   ├── data_prep.py      # load datasets, align spans to token labels
│   ├── dataset.py        # PyTorch/HF Dataset wrapper + collator
│   ├── model.py          # ModernBERT + token classification head
│   ├── train.py          # training loop
│   ├── evaluate.py       # example-level + span-level F1, ECE
│   └── infer.py          # single-triple inference
├── notebooks/            # exploratory sanity checks
├── configs/              # hyperparameters, paths (relative to repo root)
├── checkpoints/          # saved weights (gitignored)
└── results/              # metrics, comparison tables
```
