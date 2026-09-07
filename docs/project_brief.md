# LettuceDetect Reimplementation — Project Brief

**Purpose of this document:** paste this whole file into Claude Code as the first message (or save it as `PROJECT_BRIEF.md` in your repo root and tell Claude Code to read it) to get full context without re-explaining everything.

---

## 1. Project Context

- **Course requirement:** Sem 5 minor research project, 2 credits, under a college professor.
- **Format:** find a recent (2025 or later) ML/GenAI/RAG/agentic research paper and reimplement it. Requirements: public dataset if ML/DL, code availability preferred, modest hardware requirements, student-friendly but genuinely useful.
- **Continuation plan:** this project is expected to continue for 2 more semesters at 4 credits each, then a 6-credit major project in the final semester. Whatever gets built in sem 5 should be extensible, not a one-off.
- **Chosen paper:** finalized on **LettuceDetect**.

---

## 2. The Paper

**Title:** LettuceDetect: A Hallucination Detection Framework for RAG Applications
**Authors:** Ádám Kovács, Gábor Recski (KR Labs / TU Wien)
**Published:** February 2025 — arXiv:2502.17125
**Official code:** github.com/KRLabsOrg/LettuceDetect (MIT license, also `pip install lettucedetect`)
**Official models:** huggingface.co/KRLabsOrg/lettucedect-base-modernbert-en-v1 and lettucedect-large-modernbert-en-v1

### The problem it solves
RAG (Retrieval-Augmented Generation) systems still hallucinate even when given correct source documents — the LLM's answer can contain claims not actually supported by the retrieved context. Existing detectors are either LLM-based (accurate but slow and expensive) or older encoder-based models (fast but limited by short context windows, e.g. BERT's 512-token cap).

### The core idea
Frame hallucination detection as **token-level binary classification**. Given a (context, question, answer) triple, classify every token in the answer as:
- `0` = supported by the context
- `1` = hallucinated (not supported)

Built on **ModernBERT**, which supports sequences up to 8,192 tokens (vs BERT's 512), so it can handle realistically long RAG contexts. This is the paper's key technical enabler — earlier encoder-based approaches physically couldn't see enough of the context to judge groundedness.

### Architecture (Figure 2 in the paper)
```
[CLS] Context [SEP] Question [SEP] Answer
        ↓
   Tokenizer (AutoTokenizer)
        ↓
   ModernBERT backbone (base: 150M params, large: 396M params)
        ↓
   Token classification head (AutoModelForTokenClassification)
        ↓
   Per-token probability of "hallucinated"
        ↓
   Span-level output = consecutive hallucinated tokens above 0.5 threshold, merged
```
Context and question tokens are **masked from the loss** (label = -100). Only answer tokens are labeled 0/1 and contribute to training.

### Training recipe (exactly as published — no guesswork needed)
| Hyperparameter | Value |
|---|---|
| Base model | ModernBERT-base (150M) or ModernBERT-large (396M) |
| Optimizer | AdamW |
| Learning rate | 1e-5 |
| Weight decay | 0.01 |
| Epochs | 6 |
| Batch size | 8 (paper's setting, on an A100 — see Section 5 below for your GPU) |
| Max sequence length | 4,096 tokens (paper's setting) |
| Padding | Dynamic, via `DataCollatorForTokenClassification` |
| Precision | Not specified as mixed precision in the paper; use fp16/bf16 yourself to save memory |
| Checkpoint selection | Best token-level F1 on validation, saved via `safetensors` |

### Dataset: RAGTruth
- 18,000 span-annotated examples across three tasks: question answering (from MS MARCO), data-to-text generation (from Yelp Open Dataset), and news summarization (from CNN/Daily Mail).
- Each example has one response from each of 6 LLMs (GPT-4-0613, Mistral-7B-Instruct, Llama-2-7b-chat, Llama-2-13B-chat, and others) — so 6 responses per underlying sample.
- Human-annotated spans marked as hallucinated, with categories (Evident Conflict, Subtle Conflict, Evident/Subtle Introduction of Baseless Information) — **the paper collapses these into a single binary label** for training; you should do the same for the core reimplementation.
- Mean answer token length: 801, median: 741, min: 194, max: 2,632. This tells you 2,048 tokens covers the large majority of examples.
- Source: search "RAGTruth dataset github" — released by Niu et al. 2024, typically distributed as JSON with context/question/response/annotation fields.

### Evaluation metrics (both from the paper, Tables 2 and 3)
1. **Example-level F1**: binary — does the answer contain a hallucination at all? Aggregate token predictions up to one label per example. Reported per task (QA, data-to-text, summarization) and overall.
2. **Span-level F1**: character-level overlap between predicted and gold hallucinated spans. Note: the paper states the original RAGTruth code did not include this evaluation, so the authors implemented it themselves — your numbers not matching exactly is expected and defensible, not a bug.

### Published baselines you can compare against without reimplementing them
From Table 2 (example-level F1, overall): GPT-3.5-turbo prompting 52.9, GPT-4-turbo prompting 63.4, SelfCheckGPT 58.8, fine-tuned Llama-2-13B 78.7, RAG-HAT 83.9, Luna 65.4, **lettucedetect-base 76.07, lettucedetect-large 79.22**.

Your target: get reasonably close to the base model's 76.07 F1. You are not expected to beat RAG-HAT (a fine-tuned 8B LLM) with a 150M encoder — the paper doesn't either, and says so explicitly.

---

## 3. Scope for Sem 5 (2 credits, ~16 weeks)

**In scope:**
- Reimplement the token-classification pipeline from scratch (own code, not just calling their pip package) for ModernBERT-base.
- Train and evaluate on RAGTruth, at minimum on the QA subtask, ideally all three tasks if time and compute allow.
- Reproduce example-level F1 as the primary metric; attempt span-level F1 as a stretch goal.
- Compare your numbers against the published baselines in Tables 2/3 (no need to rerun their models — cite their numbers).
- Document deviations honestly (sequence length truncation, batch size, hardware) as legitimate scope decisions, not hidden shortcuts.

**Explicitly out of scope for sem 5 (save for sem 6+):**
- Multilingual extension.
- ModernBERT-large training (unless GPU access improves).
- New datasets beyond RAGTruth.
- Building a production web demo (nice-to-have, not required).

---

## 4. Suggested Repo Structure

```
lettucedetect-reimpl/
├── PROJECT_BRIEF.md              # this file
├── data/
│   ├── raw/                      # downloaded RAGTruth JSON
│   └── processed/                # tokenized + label-aligned datasets (cached)
├── src/
│   ├── data_prep.py               # load RAGTruth, align spans to token labels
│   ├── dataset.py                 # PyTorch Dataset / HF datasets wrapper
│   ├── model.py                   # ModernBERT + AutoModelForTokenClassification setup
│   ├── train.py                   # training loop (HF Trainer or custom)
│   ├── evaluate.py                # example-level + span-level F1
│   └── infer.py                   # run inference on a single (context, question, answer) triple
├── notebooks/
│   └── 01_explore_data.ipynb      # sanity-check label alignment on a handful of examples
├── configs/
│   └── base_config.yaml           # hyperparameters, paths, seq length, batch size
├── checkpoints/                   # saved model weights (gitignored, large files)
├── results/
│   ├── metrics.json               # your F1 scores vs paper's
│   └── comparison_table.md        # your results next to Table 2/3 baselines
└── requirements.txt
```

---

## 5. Step-by-Step Implementation Plan

### Phase 1 — Setup and data (Week 1-2)
1. Set up environment: `transformers`, `torch`, `datasets`, `accelerate`, `scikit-learn` for metrics.
2. Download RAGTruth. Inspect the raw JSON structure — confirm field names for context, question, response, and annotation spans.
3. Write `data_prep.py`: for each example, construct the input sequence `[CLS] context [SEP] question [SEP] answer`, tokenize with `AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")` using `return_offsets_mapping=True`.
4. **Label alignment (the trickiest part):** for each answer token, use its character offset to check whether it falls inside any annotated hallucinated span. Label context/question tokens as -100. Test this on 10 hand-picked examples and manually verify the labels look right before running it on the full dataset.
5. Cache the tokenized, labeled dataset to disk so you don't repeat this every run.

### Phase 2 — Model and training loop (Week 3-4)
6. Load `AutoModelForTokenClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=2)`.
7. Set up `DataCollatorForTokenClassification` for dynamic padding.
8. Write the training loop (Hugging Face `Trainer` is fine and saves you boilerplate) with the hyperparameters from Section 2 above.
9. **Adjust for your hardware** (you're likely on a free/Colab-Pro GPU, not an A100):
   - Truncate max sequence length to 2,048 instead of 4,096 (covers the large majority of examples per the token-length stats above).
   - If you hit OOM: drop batch size to 2-4 and use gradient accumulation to reach an effective batch size of 8.
   - Use fp16 or bf16 mixed precision.
   - Document this deviation explicitly in your report — it's expected, not a flaw.
10. Run a short training smoke test (e.g. 100 examples, 1 epoch) before committing to a full run, to catch bugs cheaply.

### Phase 3 — Full training (Week 5-7)
11. Train on the QA subtask first (smallest, fastest feedback loop).
12. Once QA works end-to-end, expand to data-to-text and summarization if time allows.
13. Save best checkpoint by validation token-level F1.

### Phase 4 — Evaluation (Week 8-9)
14. Implement example-level F1: aggregate token predictions to one label per example, compute precision/recall/F1 with `sklearn.metrics`.
15. Implement span-level F1: character-overlap matching between predicted and gold spans. Expect to spend real time here — handle edge cases like partially overlapping spans and adjacent spans.
16. Build the comparison table against Tables 2/3 baselines.

### Phase 5 — Extension / stretch goals (Week 10-12, if ahead of schedule)
17. Try ModernBERT-large if GPU allows.
18. Error analysis: which task (QA / data-to-text / summarization) does your model struggle with most, and why — this maps directly to what the paper itself found (weaker performance on summarization in Table 2).
19. Small ablation: does truncating to 1024 vs 2048 vs full length meaningfully change F1? This is a legitimate, cheap experiment that produces a real result for your report.

### Phase 6 — Write-up (Week 13-16)
20. Report: paper summary, your architecture/pipeline, hyperparameters and any deviations, results table vs published baselines, error analysis, limitations, and a proposed sem-6 continuation (see Section 7).

---

## 6. Known Pitfalls (from earlier analysis — read before you start)

1. **Silent label-alignment bugs.** If tokenizer offsets and annotation character spans are misaligned, training will run without errors but produce a garbage model. Sanity-check on hand-verified examples first.
2. **OOM on long sequences.** Expect this on your first real training run. The fixes are: shorter max length, smaller batch size + gradient accumulation, base not large model.
3. **ModernBERT's speed advantage depends on FlashAttention**, which needs an Ampere-or-newer GPU. On an older GPU (e.g. a T4, which is Turing), it'll still run correctly but via a slower fallback path — don't be surprised if it's not as fast as the paper implies.
4. **Span-level metric has no reference implementation** from the original RAGTruth authors — you're implementing this metric yourself, so don't expect an exact match to the paper's numbers.
5. **Class imbalance.** Most answer tokens are supported (label 0); hallucinated tokens (label 1) are the minority class. Watch for this in your metrics — accuracy alone will look artificially high. F1 is the right metric for a reason.

---

## 7. Continuation Ideas for Sem 6-8 (don't build these now, just keep in mind for the plan you show your professor)

- **Sem 6 (4 credits):** multilingual extension (RAGTruth is English-only; explore Hindi or another language with available RAG QA data), or a systematic ablation on sequence length / model size trade-offs, or out-of-distribution generalization testing (does a model trained on RAGTruth's LLMs generalize to hallucinations from a newer LLM not in the training set?).
- **Sem 7 (4 credits):** sentence-level vs token-level detection comparison, or integrate the detector into a live RAG pipeline as a real-time guardrail with latency benchmarking.
- **Sem 8 (6 credits, major project):** a full deployable tool — RAG pipeline plus hallucination detector plus a simple UI, evaluated with a small user study.

---

## 8. What to tell Claude Code first

Suggested opening prompt once you're in Claude Code with this file in the repo:

> "Read PROJECT_BRIEF.md. Let's start with Phase 1: set up the repo structure, then help me write data_prep.py to load RAGTruth and align the character-level hallucination spans to token-level labels using ModernBERT's tokenizer. Let's test the alignment on a handful of hand-picked examples before running it on the full dataset."
