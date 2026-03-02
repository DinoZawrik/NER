# Named Entity Recognition — Three Approaches Compared

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%9C%93-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%E2%9C%93-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-%23FFD21E)
![Dataset](https://img.shields.io/badge/Dataset-CoNLL--2003-lightgrey)

Three NER models implemented on CoNLL-2003 — from classical CRF to BERT fine-tuning.

---

## Results

| Model | F1 (micro) | PER | ORG | LOC | MISC |
|-------|-----------|-----|-----|-----|------|
| **BERT** (bert-base-cased) | **0.88** | 0.95 | 0.85 | 0.90 | 0.72 |
| CRF | 0.80 | 0.83 | 0.73 | 0.85 | 0.76 |
| BiLSTM-CRF | 0.74 | 0.72 | 0.70 | 0.82 | 0.72 |

Trained and evaluated on [CoNLL-2003](https://huggingface.co/datasets/conll2003) English split.

---

## Models

### CRF (Conditional Random Field)
Classical sequence labeling. Features: word shape, prefix/suffix, capitalization, POS tags (via NLTK), neighboring word context.

### BiLSTM-CRF
Bidirectional LSTM encodes the sentence, a CRF decoding layer enforces valid BIO tag transitions. **Viterbi algorithm implemented from scratch** in PyTorch.

### BERT fine-tuning
`bert-base-cased` fine-tuned with a token classification head (TensorFlow/Keras). Handles subword tokenization with first-subword label propagation.

---

## Example Output

```
Input:   Barack Obama visited Paris last Tuesday.

BERT:    Barack[B-PER] Obama[I-PER] visited Paris[B-LOC] last Tuesday[B-MISC].
CRF:     Barack[B-PER] Obama[I-PER] visited Paris[B-LOC] last Tuesday[O].
BiLSTM:  Barack[B-PER] Obama[I-PER] visited Paris[B-LOC] last Tuesday[O].
```

---

## Project Structure

```
NER/
├── data/
│   └── raw/
│       ├── eng.train      # 14 987 sentences
│       ├── eng.testa      # 3 466 sentences (validation)
│       └── eng.testb      # 3 684 sentences (test)
├── models/
│   ├── bilstm_crf_model.pth
│   └── crf_model.pkl
├── notebooks/
│   ├── ner_preprocessing_and_models.ipynb
│   └── ner_model_implementation.ipynb
├── scripts/
│   ├── utils.py                 # shared data loading, vocab, BiLSTM-CRF model
│   ├── ner_model_comparison.py  # train + evaluate CRF and BiLSTM-CRF
│   ├── ner_bert_kaggle.py       # BERT fine-tuning (TensorFlow)
│   └── error_analysis.py        # per-sentence error comparison across models
├── outputs/
│   └── all_models_output.txt    # training logs and classification reports
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/DinoZawrik/NER.git
cd NER
pip install -r requirements.txt
```

Download CoNLL-2003 data (requires LDC license) and place files in `data/raw/`.

**Train CRF and BiLSTM-CRF:**
```bash
python scripts/ner_model_comparison.py
```

**BERT fine-tuning** was run on Kaggle (GPU required). See `scripts/ner_bert_kaggle.py`.

**Error analysis across all three models:**
```bash
python scripts/error_analysis.py
```

---

## Technical Highlights

- **Viterbi decoding** — implemented from scratch in PyTorch for BiLSTM-CRF
- **Subword alignment** — BERT tokenizer produces multiple subword tokens per word; first-subword strategy maps labels back to original word boundaries
- **CRF contextual features** — word shape, capitalization, prefix/suffix, POS tags, window context ±2 words
- **Three frameworks in one project** — PyTorch (BiLSTM-CRF), TensorFlow (BERT), scikit-learn (CRF)
- **Comparative error analysis** — `error_analysis.py` shows per-sentence disagreements across all three models
