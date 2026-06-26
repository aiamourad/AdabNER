# A<sub>DAB</sub>NER — Arabic Digital Archive Books with Nested Entity Recognition

[![ACL 2026](https://img.shields.io/badge/ACL_2026-Long_Paper-lightgrey?style=flat-square)](https://aclanthology.org/2026.acl-long.1541/)
[![Paper](https://img.shields.io/badge/Paper-aclanthology-lightgrey?style=flat-square)](https://aclanthology.org/2026.acl-long.1541/)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo_(coming_soon)-lightgrey?style=flat-square)](https://doi.org/10.5281/zenodo.19468385)
[![Code](https://img.shields.io/badge/Code-GitHub-lightgrey?style=flat-square&logo=github)](https://github.com/aiamourad/AdabNER)
[![License](https://img.shields.io/badge/Data_License-Non--commercial_Research-lightgrey?style=flat-square)]()

**Aya Mourad** · **Mustafa Jarrar**

---

## At a Glance

| Books | Genres | Tokens | Entity mentions | Entity types | Nested | IAA κ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 138 | 10 | 876K | 78,530 | 21 | 18.96% | 0.938 |

---

## Abstract

Most studies on Arabic Named Entity Recognition (NER) have focused on news texts and social media posts, while the large and rich corpus of literary Arabic books has been underrepresented. We introduce AdabNER, the first large-scale nested NER dataset for Modern Standard Arabic (MSA) literary texts, comprising the first 6,000 words annotated from each of 138 books spanning ten literary genres, including history, biography, literary criticism, and travel literature, and covering works from the 1880s to the 2020s. The corpus comprises about 876K tokens, manually annotated using a nested 21 entity tag annotation scheme, yielding 78,530 entity mentions, 18.96% of which are nested. We fine-tuned five pre-trained Arabic BERT encoders in two settings: stratified and leave-book-out, achieving F1 scores of 0.86 and 0.83 with AraBERTv2, respectively. We also evaluated five large language models through few-shot in-context learning, including open-source models and the closed-source Gemini 3 Pro, with Gemini 3 Pro achieving the highest LLM F1 score of 0.59. Supervised results degraded under out-of-domain evaluation; however, joint multi-domain training reduced this gap to less than a 1% F1 loss, demonstrating that domain-diverse training data is key to robust Arabic NER, though broader validation beyond the experiments reported is needed. AdabNER and its annotation guidelines are publicly available at https://doi.org/10.5281/zenodo.19468385.

---

## Dataset

> **The dataset is currently undergoing final processing and will be released shortly.**
> For non-commercial research use only — see [License](#license).

| | |
|---|---|
| ADaBNER corpus + guidelines | [doi.org/10.5281/zenodo.19468385](https://doi.org/10.5281/zenodo.19468385) |
| Wojood (OOD experiments) | [sina.birzeit.edu/wojood](https://sina.birzeit.edu/wojood/) |

Place splits under `data/adabner/` and `data/wojood/` before running experiments.

---

## Corpus

### Construction

| | |
|---|---|
| Source | Hindawi Library digital collection |
| Sampling | 1 book per genre per decade; first 6,000 words after removing front matter |
| Taxonomy | 18 OntoNotes types + `OCC` / `CURR` / `UNIT` → **21 types** |
| Nesting framework | Wojood & ACE nested annotation framework |

### Genres

Biographies · Geography · History · Literary Criticism · Literature · Novels · Philosophy · Politics · Social Sciences · Travel Literature

### Statistics

| | |
|---|---|
| Books | 138 (1 per genre per decade) |
| Sentences | 26,162 · ~189 per book · mean 33.5 words |
| Tokens | ~876K |
| Entity mentions | 78,530 |
| Nested entities | 14,082 **(18.96%)** |
| Max nesting depth | 3 |
| Same-type nesting | <1% (handled with `B-ORG`, `B-ORG_2` indexing) |

### Entity Types

Highly long-tailed distribution. People and places dominate; rare types are handled with focal loss (α=0.75, γ=1.0).

| Type | Count | Type | Count | Type | Count |
|:---|---:|:---|---:|:---|---:|
| PERS | 15,495 | LOC | 5,047 | UNIT | 761 |
| GPE | 13,883 | ORG | 3,759 | QUANTITY | 758 |
| NORP | 7,065 | ORDINAL | 2,987 | CURR | 742 |
| OCC | 6,499 | TIME | 2,143 | MONEY | 718 |
| DATE | 6,495 | WORK_OF_ART | 1,927 | LAW | 296 |
| CARDINAL | 5,321 | FAC | 1,794 | PERCENT | 247 |
| | | EVENT | 1,315 | PRODUCT | 220 |
| | | LANGUAGE | 1,058 | | |

### Annotation Challenges

- **Colonial-era ambiguity** — distinguishing `LOC` vs `GPE` for proto-states.
- **Ottoman–Persian lexicon** — tagging titles such as *Jokhdar*, *Sipahi*.
- **Dating systems** — Coptic years and months.
- **Transliterated entities** — non-Arab names lack a standardised Arabic spelling.

### Inter-Annotator Agreement

Computed by three annotators on ~5% double-annotated samples (~83K tokens) using Cohen's κ and entity-level F₁.

| Cohen's κ (overall) | F₁ (overall) |
|:---:|:---:|
| 0.938 | 0.923 |

Per-genre κ: **0.901** (Novels) → **0.982** (Literature). Annotation guidelines were refined iteratively and are released with the corpus.

---

## Model & Training

**Architecture** · BERT encoder → dropout (p=0.1) → linear classifier with **sigmoid** activation. 43 BIO labels (21 types × B/I + O) — each token gets independent per-label probabilities, enabling multi-label nested prediction. Same-type nesting (<1%) uses indexed labels (`B-ORG`, `B-ORG_2`).

**Training** · AdamW · lr=6e-5 · batch=16 · max_len=512 · focal loss (α=0.75, γ=1.0) · 50 epochs, early stopping patience=5 on val macro-F₁ · threshold=0.5 · 3 seeds [42, 1, 123] · 3× NVIDIA RTX A6000 (48 GB)

**ICL** · 5-shot · stratified diverse example selection · JSON output · temperature=0 · vLLM TP=8 on A100-80GB (open-source) · Vertex AI async (Gemini, Qwen 235B)

---

## Repository Structure

```
bert/   train_adabner.py          # BERT: 5 encoders × 2 splits × 3 seeds
llm/    eval_cohere.py            # ICL: aya-expanse-32b, c4ai-command-r (vLLM)
        eval_gemini.py            # ICL: Gemini 3 Pro (Vertex AI async)
        eval_qwen.py              # ICL: Qwen2.5-72B (vLLM)
        eval_qwen_235b.py         # ICL: Qwen3-235B (Vertex AI async)
ood/    eval_wojood.py            # Train Wojood → zero-shot transfer to AdabNER
        eval_adabner_on_wojood.py # AdabNER model → zero-shot transfer to Wojood
        mmd_analysis.py           # MMD domain shift analysis
joint/  train_joint.py            # Joint training: AdabNER + Wojood
src/    model.py  dataset.py  metrics.py  preprocessing.py  llm_utils.py
tests/  test_*.py
data/   adabner/  wojood/
```

---

## Setup

```bash
pip install -r requirements.txt
```

All scripts run from the **repository root**:

```bash
python bert/train_adabner.py          # BERT experiments
python llm/eval_gemini.py             # LLM ICL
python ood/eval_wojood.py             # OOD evaluation
python joint/train_joint.py           # Joint training
```

Results written to `results/`.

---

## Results

### BERT-based Models (AdabNER)

| Model | Split | Micro P | Micro R | Micro F₁ | Macro F₁ |
|:---|:---|:---:|:---:|:---:|:---:|
| **AraBERTv2** | Stratified | 0.85±.003 | 0.87±.003 | **0.86**±.003 | **0.83**±.006 |
| **AraBERTv2** | Leave-Book-Out | 0.82±.014 | 0.85±.004 | **0.83**±.006 | **0.81**±.007 |
| ARBERTv2 | Stratified | 0.82±.007 | 0.85±.002 | 0.83±.004 | 0.81±.006 |
| ARBERTv2 | Leave-Book-Out | 0.79±.019 | 0.83±.011 | 0.81±.005 | 0.78±.006 |
| AraBERTv1 | Stratified | 0.80±.007 | 0.82±.003 | 0.81±.005 | 0.78±.011 |
| CAMeLBERT | Stratified | 0.79±.006 | 0.81±.006 | 0.80±.006 | 0.78±.008 |
| ArBERT | Stratified | 0.76±.011 | 0.80±.006 | 0.79±.008 | 0.75±.015 |

### In-Context Learning — 5-shot (AdabNER)

| Model | Split | Micro P | Micro R | Micro F₁ | Macro F₁ |
|:---|:---|:---:|:---:|:---:|:---:|
| **Gemini 3 Pro** | Leave-Book-Out | 0.55±.009 | 0.64±.020 | **0.59**±.007 | **0.57**±.005 |
| Gemini 3 Pro | Stratified | 0.59±.007 | 0.49±.040 | 0.53±.026 | 0.51±.023 |
| Qwen3-235B | Leave-Book-Out | 0.30±.012 | 0.53±.003 | 0.38±.009 | 0.35±.021 |
| Qwen2.5-72B | Leave-Book-Out | 0.30±.020 | 0.53±.004 | 0.38±.017 | 0.35±.026 |
| aya-expanse-32b | Stratified | 0.32±.013 | 0.44±.013 | 0.37±.006 | 0.32±.009 |
| c4ai-command-r | Stratified | 0.22±.008 | 0.38±.007 | 0.27±.004 | 0.24±.003 |

> LLMs struggle with nested Arabic: fine-tuned encoders dominate (F₁ = **0.86**) vs. the best LLM (F₁ = **0.59**).

### Out-of-Domain Transfer & Joint Training (AraBERTv2 · Micro F₁)

| Setting | Test Set | Micro F₁ |
|:---|:---|:---:|
| AdabNER → AdabNER *(in-domain)* | AdabNER | **0.860** |
| Joint → AdabNER | AdabNER | 0.853 |
| Wojood → AdabNER *(zero-shot)* | AdabNER | 0.590 |
| Wojood → Wojood *(in-domain)* | Wojood | **0.920** |
| Joint → Wojood | Wojood | 0.921 |
| AdabNER → Wojood *(zero-shot)* | Wojood | 0.660 |

> Joint training solves domain collapse. Cross-domain transfer drops from **0.86** to **0.66** due to temporal and lexical shifts; joint training recovers cross-domain performance to within **<1% F₁** of in-domain on both datasets.

All results averaged over 3 seeds. Full tables in the paper.

---

## Citation

```bibtex
@inproceedings{mourad-jarrar-2026-adabner,
    title     = "{A}dab{NER}: {A}rabic Digital Archive Books with Nested Entity Recognition",
    author    = "Mourad, Aya and Jarrar, Mustafa",
    booktitle = "Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month     = jul,
    year      = "2026",
    address   = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2026.acl-long.1541/",
    pages     = "33382--33396",
    ISBN      = "979-8-89176-390-6",
}
```

---

## Acknowledgments

This work was supported by the Sorbonne Center for Artificial Intelligence (SCAI) and the SOUND.AI project, funded by the European Union's Horizon programme.

---

## License

**The ADaBNER dataset is for non-commercial research use only.**

The corpus is derived from the [Hindawi Books](https://www.hindawi.org/) digital collection. The Hindawi Foundation has granted explicit permission to publish this dataset for non-commercial research purposes, with the following conditions:

- Use for **non-commercial research purposes only**
- **Do not redistribute** to third parties
- **Acknowledge the Hindawi Foundation** in any publication that uses the dataset

The code in this repository is released under the **MIT License**.