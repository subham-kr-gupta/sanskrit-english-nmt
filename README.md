# Sanskrit → English Neural Machine Translation

A custom attention-based sequence-to-sequence system for translating Sanskrit into English,
built from scratch for the Natural Language Understanding course (Assignment 2).

## Overview

Sanskrit is a low-resource, morphologically rich language. This project implements a compact
NMT model — a **bidirectional-GRU encoder + Bahdanau (additive) attention + GRU decoder** with
subword (SentencePiece BPE) tokenization and beam-search-capable decoding. No pretrained
translation model is used; the entire system is trained from scratch on the provided 10,000
parallel sentence pairs.

The design is intentionally small (~20M parameters, ~2s inference for the 1000-sentence test
set) to balance translation quality against the assignment's efficiency scoring.

## Repository contents

| File | Description |
|------|-------------|
| `sanskrit_en_nmt.ipynb` | Complete, self-contained training + evaluation notebook |
| `NMT_Report.pdf` | Full report (architecture, training, results, error analysis) |
| `submission.csv` | Predicted English translations for the test set (`Source_id`, `Sentence_en`) |
| `train_sa_10000.csv`, `train_en_10000.csv` | Training pairs (10,000 sentences) |
| `dev_sa_1000.csv`, `dev_en_1000.csv` | Validation pairs (1,000 sentences) |
| `test_sa_1000.csv`, `test_en_1000.csv` | Test pairs (1,000 sentences) |

The Sanskrit (`*_sa_*`) and English (`*_en_*`) files are aligned by a shared `Source_id`
column. The notebook joins them automatically.

## How to run

1. The six dataset files are included in this repo alongside the notebook (the loader
   auto-detects names such as `train_sa_10000.csv`, `dev_en_1000.csv`, `test_sa_1000.csv`, …).
2. Open `sanskrit_en_nmt.ipynb` and run all cells top to bottom. Cell 0 installs all
   dependencies. A GPU is strongly recommended (training ~15–40 min on an A100).
3. The notebook writes `submission.csv` and prints BLEU, BERTScore, inference time, and the
   parameter count.

> **BERTScore offline note:** the metric uses `roberta-large` internally. If the machine has no
> internet, pre-download it into the Hugging Face cache and set
> `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` at the top of the notebook.

## Results (greedy decoding)

| Split | Corpus BLEU | Sentence BLEU | BERTScore F1 |
|-------|-------------|---------------|--------------|
| Dev   | 0.0678      | 0.0773        | 0.2146       |
| Test  | 0.0659      | 0.0816        | 0.2145       |

- **Parameters:** 20,378,016
- **Inference time:** ~2.15 s for 1000 test sentences (NVIDIA A100)

## Model summary

- **Tokenization:** SentencePiece BPE, 4000 tokens per language
- **Encoder:** 2-layer bidirectional GRU, embedding 256, hidden 512
- **Attention:** additive / Bahdanau, masked over padding
- **Decoder:** 1-layer GRU with attention context
- **Training:** Adam (lr 1e-3, ReduceLROnPlateau), dropout 0.4, label smoothing 0.1,
  gradient clipping 1.0, teacher forcing 0.5, early stopping on dev BLEU

## Disclosure

No pretrained model is used for translation. The only pretrained network is `roberta-large`,
used solely by the `bert-score` library to compute the BERTScore evaluation metric.
