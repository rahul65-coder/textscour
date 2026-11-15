# TEXTSCOUR Document Quality Scoring Rubric (`docs/SCORING.md`)

This document defines the mathematical formulation, heuristic weights, edge cases, and 15 canonical worked examples for the TEXTSCOUR document quality scoring rubric (implemented in `scour_core::pipeline::score_doc`).

---

## 1. Quality Score Formula (MASTER_PLAN Section A.4 & INTERFACES.md)

Quality score `score` is a real number in $[0.0, 10.0]$ rounded to one decimal place, where higher indicates a cleaner, higher-value training document for pretraining / SFT datasets.

### Component Terms

1. **Chunk Quality Term ($q$)** (Weight: 40%):
   $$q = \frac{\sum_{i=1}^N \text{chars}_i \cdot (1 - g_i)}{\sum_{i=1}^N \text{chars}_i}$$
   Over all *original* document chunks $i \in \{1, \dots, N\}$.
   - If chunk $i$ was dropped ($\text{keep} = \text{false}$): $g_i = 1.0$.
   - If chunk $i$ was kept ($\text{keep} = \text{true}$): $g_i = \max(\text{rule\_garbage}_i, \text{promo\_chunk}_i, \text{gpu\_p}_i) \in [0.0, 1.0]$.
   - $q \in [0.0, 1.0]$.

2. **Promo Preservation Term ($p\_term$)** (Weight: 20%):
   $$p\_term = 1.0 - \min\left(1.0, \frac{\text{promo\_chars\_removed} / \text{orig\_chars}}{0.25}\right)$$
   - Documents with 0% promo removed achieve $p\_term = 1.0$.
   - Documents losing $\ge 25\%$ of original characters to promo/noise stripping reach $p\_term = 0.0$.

3. **URL Cleanliness Term ($url\_term$)** (Weight: 10%):
   $$\text{urls\_per\_1k} = \frac{\text{remaining\_urls} \times 1000}{\max(1.0, L)}$$
   $$url\_term = 1.0 - \min\left(1.0, \frac{\text{urls\_per\_1k}}{5.0}\right)$$
   Where $L$ is the final Unicode character length. Documents with $> 5$ URLs per 1,000 characters receive $url\_term = 0.0$.

4. **Document Length Term ($len\_term$)** (Weight: 15%):
   $$len\_term = \min\left(1.0, \frac{\log_{10}(1.0 + L / 200.0)}{\log_{10}(11.0)}\right)$$
   - When $L = 2000$, $1 + 2000/200 = 11$, and $len\_term = 1.0$.
   - Prevents micro-fragments from achieving top scores while gracefully saturating at standard document lengths.

5. **Language Validity Term ($lang\_term$)** (Weight: 15%):
   $$lang\_term = \begin{cases} \text{clamp}(\text{lang\_conf}, 0.0, 1.0) & \text{if language is allowed} \\ 0.0 & \text{otherwise} \end{cases}$$

6. **Orphan References Penalty**:
   $$\text{penalty} = \min(1.0, 0.10 \times \text{orphan\_refs})$$
   Triggered when a kept chunk begins with a dangling back-reference ("as mentioned above", "the above table", etc.) whose preceding original chunk was discarded as noise.

### Combined Formula
$$\text{raw\_score} = 10.0 \times \left(0.40 \cdot q + 0.20 \cdot p\_term + 0.10 \cdot url\_term + 0.15 \cdot len\_term + 0.15 \cdot lang\_term\right) - \text{penalty}$$
$$\text{score} = \text{round}\left(\text{clamp}(\text{raw\_score}, 0.0, 10.0), 1\right)$$

### Clean Document Criterion
A document is classified as `clean = true` if and only if:
1. $\text{score} \ge \text{clean\_threshold}$ (default: $6.0$)
2. $L \ge \text{min\_doc\_chars}$ (default: $200$)
3. Detected language $\in \text{allowed\_langs}$
4. $\text{pii\_dropped} = \text{false}$
5. $\text{rebuilt.text}$ is non-empty and non-whitespace

---

## 2. Fifteen Canonical Worked Examples

These 15 canonical test cases are mirrored identically in `tests/score_cases.rs`.

| Case | Description | Chunks & Scores ($g_i$, keep) | Promo Cut / Orig | URLs / Final Len | Lang (Allowed, Conf) | Orphan Refs | Score | Clean Verdict |
|---|---|---|---|---|---|---|---|---|
| **1** | **Perfect Long Document** | 1 chunk: 2000 ch, $g=0.0$, keep | 0 / 2000 ch | 0 / 2000 ch | Yes, 1.00 | 0 | **10.0** | Clean |
| **2** | **Pure Garbage Document** | 1 chunk: 500 ch, $g=1.0$, drop | 500 / 500 ch | 0 / 0 ch | Yes, 0.80 | 0 | **2.2** | Garbage |
| **3** | **Wrong Language Doc** | 1 chunk: 1000 ch, $g=0.0$, keep | 0 / 1000 ch | 0 / 1000 ch | **No**, 0.90 | 0 | **8.1** | Garbage (`wrong_lang`) |
| **4** | **Moderate Promo Stripped** | 2 chunks: 500 ch ($g=0$), 500 ch ($g=0$) | 100 / 1000 ch (10%) | 0 / 900 ch | Yes, 1.00 | 0 | **8.8** | Clean |
| **5** | **Heavy Promo Stripped** | 2 chunks: 500 ch ($g=0$), 500 ch ($g=1$, drop) | 500 / 1000 ch (50%) | 0 / 500 ch | Yes, 0.95 | 0 | **5.2** | Garbage (`low_score`) |
| **6** | **High URL Density** | 1 chunk: 1000 ch, $g=0.0$, keep | 0 / 1000 ch | 5 / 1000 ch (5/1k) | Yes, 1.00 | 0 | **8.6** | Clean |
| **7** | **Moderate URL Density** | 1 chunk: 1000 ch, $g=0.0$, keep | 0 / 1000 ch | 2 / 1000 ch (2/1k) | Yes, 1.00 | 0 | **9.2** | Clean |
| **8** | **Short Text (L=100)** | 1 chunk: 100 ch, $g=0.0$, keep | 0 / 100 ch | 0 / 100 ch | Yes, 1.00 | 0 | **8.8** | Garbage (`too_short`) |
| **9** | **Micro-Text (L=20)** | 1 chunk: 20 ch, $g=0.0$, keep | 0 / 20 ch | 0 / 20 ch | Yes, 0.80 | 0 | **8.3** | Garbage (`too_short`) |
| **10** | **Orphan Ref Penalty** | 2 chunks: 500 ch ($g=1$, drop), 500 ch ($g=0$) | 500 / 1000 ch | 0 / 500 ch | Yes, 1.00 | 3 (-0.3) | **5.0** | Garbage (`low_score`) |
| **11** | **Max Orphan Penalty** | 1 chunk: 2000 ch, $g=0.0$, keep | 0 / 2000 ch | 0 / 2000 ch | Yes, 1.00 | 12 (-1.0 cap) | **9.0** | Clean |
| **12** | **Low Lang Confidence** | 1 chunk: 2000 ch, $g=0.0$, keep | 0 / 2000 ch | 0 / 2000 ch | Yes, 0.50 | 0 | **9.2** | Clean |
| **13** | **Borderline Quality Doc** | 3 chunks: 400 ch ($g=0.3$), 400 ch ($g=0.5$), 400 ch ($g=0.2$) | 50 / 1200 ch | 2 / 1150 ch | Yes, 0.90 | 1 (-0.1) | **7.4** | Clean |
| **14** | **Degraded Multi-Flaw** | 2 chunks: 500 ch ($g=0.6$), 500 ch ($g=1.0$, drop) | 500 / 1000 ch | 4 / 400 ch (10/1k) | Yes, 0.70 | 2 (-0.2) | **2.3** | Garbage (`low_score`) |
| **15** | **Clean Hindi Article** | 2 chunks: 600 ch ($g=0.0$), 600 ch ($g=0.0$) | 0 / 1200 ch | 1 / 1200 ch | Yes, 0.98 | 0 | **9.5** | Clean |
