<div align="center">

# A<sub>DAB</sub>NER
### Arabic Digital Archive Books with Nested Entity Recognition

[![ACL 2026](https://img.shields.io/badge/ACL_2026-Long_Paper-1F3A6E?style=flat-square)](https://aclanthology.org/2026.acl-long.1541/)
[![Paper](https://img.shields.io/badge/Paper-aclanthology-C0392B?style=flat-square)](https://aclanthology.org/2026.acl-long.1541/)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo_(coming_soon)-4E6BA6?style=flat-square)](https://doi.org/10.5281/zenodo.19468385)
[![Code](https://img.shields.io/badge/Code-GitHub-24292e?style=flat-square&logo=github)](https://github.com/aiamourad/AdabNER)
[![License](https://img.shields.io/badge/Data_License-Non--commercial_Research-E67E22?style=flat-square)]()

**Aya Mourad** &nbsp;·&nbsp; **Mustafa Jarrar**

</div>

---

## At a Glance

<div align="center">
<table>
<tr>
<td align="center" width="16%">
<img src="https://img.shields.io/badge/138-1F3A6E?style=for-the-badge"/><br>
<sub><b>BOOKS · 10 GENRES</b></sub>
</td>
<td align="center" width="16%">
<img src="https://img.shields.io/badge/876K-1F3A6E?style=for-the-badge"/><br>
<sub><b>TOKENS</b></sub>
</td>
<td align="center" width="16%">
<img src="https://img.shields.io/badge/78.5K-1F3A6E?style=for-the-badge"/><br>
<sub><b>ENTITY MENTIONS</b></sub>
</td>
<td align="center" width="16%">
<img src="https://img.shields.io/badge/21-1F3A6E?style=for-the-badge"/><br>
<sub><b>ENTITY TYPES</b></sub>
</td>
<td align="center" width="16%">
<img src="https://img.shields.io/badge/18.96%25-1F3A6E?style=for-the-badge"/><br>
<sub><b>NESTED</b></sub>
</td>
<td align="center" width="16%">
<img src="https://img.shields.io/badge/0.938-1F3A6E?style=for-the-badge"/><br>
<sub><b>IAA κ</b></sub>
</td>
</tr>
</table>
</div>

---

## Abstract

Most studies on Arabic NER have focused on news texts and social media posts, while the large and rich corpus of literary Arabic books has been underrepresented. Existing Arabic NER datasets cover news, Wikipedia, and social media, and models trained on these resources collapse on literary text. We introduce **AdabNER**, the first large-scale nested NER dataset for Modern Standard Arabic (MSA) literary texts — 138 books · 10 genres · 15 decades (1880s–2020s) · 876K tokens · 21 entity types · 78,530 mentions · 18.96% nested.

We fine-tuned five Arabic BERT encoders under two splits, achieving F₁ **0.86** / **0.83** (stratified / leave-book-out) with AraBERTv2. Five LLMs evaluated via 5-shot ICL; Gemini 3 Pro reaches F₁ **0.59**. Joint multi-domain training with Wojood closes the out-of-domain gap to **<1% F₁** loss.

---

## Dataset

> **The dataset is currently undergoing final processing and will be released shortly.**  
> For non-commercial research use only — see [License](#license).

| | |
|---|---|
| AdabNER corpus + guidelines | [doi.org/10.5281/zenodo.19468385](https://doi.org/10.5281/zenodo.19468385) |
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

<div align="center">
<table>
<tr>
<td align="center">🟠&nbsp;<b>Biographies</b></td>
<td align="center">🟢&nbsp;<b>Geography</b></td>
<td align="center">🟣&nbsp;<b>History</b></td>
<td align="center">🔵&nbsp;<b>Literary Criticism</b></td>
<td align="center">🔴&nbsp;<b>Literature</b></td>
</tr>
<tr>
<td align="center">🩵&nbsp;<b>Novels</b></td>
<td align="center">🟡&nbsp;<b>Philosophy</b></td>
<td align="center">🟪&nbsp;<b>Politics</b></td>
<td align="center">⬛&nbsp;<b>Social Sciences</b></td>
<td align="center">🔷&nbsp;<b>Travel Literature</b></td>
</tr>
</table>
</div>

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

<div align="center">

![PERS](https://img.shields.io/badge/PERS-15%2C495-FFA500?style=flat-square)
![GPE](https://img.shields.io/badge/GPE-13%2C883-3B9BDC?style=flat-square)
![NORP](https://img.shields.io/badge/NORP-7%2C065-008A8A?style=flat-square)
![OCC](https://img.shields.io/badge/OCC-6%2C499-E75244?style=flat-square)
![DATE](https://img.shields.io/badge/DATE-6%2C495-8D44AC?style=flat-square)
![CARDINAL](https://img.shields.io/badge/CARDINAL-5%2C321-808080?style=flat-square)
![LOC](https://img.shields.io/badge/LOC-5%2C047-3B9BDC?style=flat-square)
![ORG](https://img.shields.io/badge/ORG-3%2C759-E67E22?style=flat-square)
![ORDINAL](https://img.shields.io/badge/ORDINAL-2%2C987-808080?style=flat-square)
![TIME](https://img.shields.io/badge/TIME-2%2C143-8D44AC?style=flat-square)
![WORK\_OF\_ART](https://img.shields.io/badge/WORK__OF__ART-1%2C927-E74C3C?style=flat-square)
![FAC](https://img.shields.io/badge/FAC-1%2C794-3498DB?style=flat-square)
![EVENT](https://img.shields.io/badge/EVENT-1%2C315-1ABC9C?style=flat-square)
![LANGUAGE](https://img.shields.io/badge/LANGUAGE-1%2C058-27AE60?style=flat-square)
![UNIT](https://img.shields.io/badge/UNIT-761-95A5A6?style=flat-square)
![QUANTITY](https://img.shields.io/badge/QUANTITY-758-95A5A6?style=flat-square)
![CURR](https://img.shields.io/badge/CURR-742-F39C12?style=flat-square)
![MONEY](https://img.shields.io/badge/MONEY-718-F39C12?style=flat-square)
![LAW](https://img.shields.io/badge/LAW-296-7F8C8D?style=flat-square)
![PERCENT](https://img.shields.io/badge/PERCENT-247-7F8C8D?style=flat-square)
![PRODUCT](https://img.shields.io/badge/PRODUCT-220-7F8C8D?style=flat-square)

</div>

Highly long-tailed distribution. People and places dominate; rare types are handled with focal loss (α=0.75, γ=1.0).

### Annotation Challenges

- **Colonial-era ambiguity** — distinguishing `LOC` vs `GPE` for proto-states.
- **Ottoman–Persian lexicon** — tagging titles such as *Jokhdar*, *Sipahi*.
- **Dating systems** — Coptic years and months.
- **Transliterated entities** — non-Arab names lack a standardised Arabic spelling.

### Inter-Annotator Agreement

Computed by three annotators on ~5% double-annotated samples (~83K tokens) using Cohen's κ and entity-level F₁.

<div align="center">
<table>
<tr>
<td align="center" width="50%">
<img src="https://img.shields.io/badge/0.938-008A8A?style=for-the-badge"/><br>
<sub><b>COHEN'S κ · OVERALL</b></sub>
</td>
<td align="center" width="50%">
<img src="https://img.shields.io/badge/0.923-008A8A?style=for-the-badge"/><br>
<sub><b>F₁ · OVERALL</b></sub>
</td>
</tr>
</table>
</div>

Per-genre κ: **0.901** (Novels) → **0.982** (Literature). Annotation guidelines were refined iteratively and are released with the corpus.

---

## Model & Training

**Architecture** &nbsp;·&nbsp; BERT encoder → dropout (p=0.1) → linear classifier with **sigmoid** activation. 43 BIO labels (21 types × B/I + O) — each token gets independent per-label probabilities, enabling multi-label nested prediction. Same-type nesting (<1%) uses indexed labels (`B-ORG`, `B-ORG_2`).

**Training** &nbsp;·&nbsp; AdamW · lr=6e-5 · batch=16 · max\_len=512 · focal loss (α=0.75, γ=1.0) · 50 epochs, early stopping patience=5 on val macro-F₁ · threshold=0.5 · 3 seeds [42, 1, 123] · 3× NVIDIA RTX A6000 (48 GB)

**ICL** &nbsp;·&nbsp; 5-shot · stratified diverse example selection · JSON output · temperature=0 · vLLM TP=8 on A100-80GB (open-source) · Vertex AI async (Gemini, Qwen 235B)

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

<div align="center">
<table>
<tr>
<th>Setting</th><th>Test Set</th><th>Micro F₁</th><th></th>
</tr>
<tr>
<td>AdabNER → AdabNER <em>(in-domain)</em></td><td>AdabNER</td>
<td align="center"><b>0.860</b></td>
<td><img src="https://img.shields.io/badge/■■■■■■■■■■■■■■■■■■■■■■■■-1F3A6E?style=flat-square&label="/></td>
</tr>
<tr>
<td>Joint → AdabNER</td><td>AdabNER</td>
<td align="center">0.853</td>
<td><img src="https://img.shields.io/badge/■■■■■■■■■■■■■■■■■■■■■■■-4E6BA6?style=flat-square&label="/></td>
</tr>
<tr>
<td>Wojood → AdabNER <em>(zero-shot)</em></td><td>AdabNER</td>
<td align="center">0.590</td>
<td><img src="https://img.shields.io/badge/■■■■■■■■■■■■■-E67E22?style=flat-square&label="/></td>
</tr>
<tr><td colspan="4"><hr/></td></tr>
<tr>
<td>Wojood → Wojood <em>(in-domain)</em></td><td>Wojood</td>
<td align="center"><b>0.920</b></td>
<td><img src="https://img.shields.io/badge/■■■■■■■■■■■■■■■■■■■■■■■■-1F3A6E?style=flat-square&label="/></td>
</tr>
<tr>
<td>Joint → Wojood</td><td>Wojood</td>
<td align="center">0.921</td>
<td><img src="https://img.shields.io/badge/■■■■■■■■■■■■■■■■■■■■■■■■-4E6BA6?style=flat-square&label="/></td>
</tr>
<tr>
<td>AdabNER → Wojood <em>(zero-shot)</em></td><td>Wojood</td>
<td align="center">0.660</td>
<td><img src="https://img.shields.io/badge/■■■■■■■■■■■■■■■-E67E22?style=flat-square&label="/></td>
</tr>
</table>
</div>

> Joint training solves domain collapse. Cross-domain transfer drops from **0.86** to **0.66** due to temporal and lexical shifts; joint training recovers cross-domain performance to within **<1% F₁** of in-domain on **both** datasets.

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

**The AdabNER dataset is for non-commercial research use only.**

The corpus is derived from the [Hindawi Books](https://www.hindawi.org/) digital collection. The Hindawi Foundation has granted explicit permission to publish this dataset for non-commercial research purposes, with the following conditions:

- Use for **non-commercial research purposes only**
- **Do not redistribute** to third parties
- **Acknowledge the Hindawi Foundation** in any publication that uses the dataset

The code in this repository is released under the **MIT License**.