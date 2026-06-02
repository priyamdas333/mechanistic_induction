# Reverse Engineering Attention Heads for Induction Behaviour detection in Small Transformers

[![Research Area](https://img.shields.io/badge/Research-Mechanistic%20Interpretability-blueviolet)](https://github.com/priyamdas333/Mechanistic_induction_Interpretability)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Experimental%20Phase-success)](#)

This repository contains a mechanistic interpretability pipeline designed to open the "black box" of small, autoregressive transformer models. Specifically, we isolate, visualize, and causally validate **Induction Heads**—the fundamental circuits responsible for in-context learning, prefix-matching, and macro algorithmic pattern recognition in large language models.

---

## 🔬 Core Research Framework

An induction head is characterized by a specific computational sub-routine: it searches for a previous occurrence of the current token $A$, looks at the subsequent token $B$, and copies that information to predict $B$ when $A$ recurs ($[A][B] \dots [A] \rightarrow \text{predict } [B]$).

We investigate two distinct methodologies to identify and track these heads, exposing a fascinating discrepancy between **geometric correlation** and **functional causality**.

### Methodology 1: Attention Pattern Analysis (Correlation)
We track the attention matrices $A_{l,h}$ across all layers $l$ and heads $h$. An induction head is mathematically hypothesized to exhibit a *shifted-diagonal* attention pattern, consistently looking back exactly $N$ tokens (where $N = \text{pattern\_len}$).
$$\text{Induction Score}(l, h) = \frac{1}{|S|} \sum_{i \in S} A_{l,h}[i, i - \text{pattern\_len}]$$

### Methodology 2: Causal Ablation (Causality)
We perform intervention experiments by zeroing out the direct output vector of specific attention heads during the forward pass ($O_{l,h} \leftarrow 0$) and measuring the downstream cross-entropy loss delta ($\Delta \mathcal{L}$) or accuracy drop on a sequence repetition task.
$$\Delta \mathcal{L}(l,h) = \mathcal{L}(\mathbf{x}; \text{Ablated}_{l,h}) - \mathcal{L}(\mathbf{x}; \text{Baseline})$$

---

## 📊 Empirical Observations & Discrepancies

When evaluating a toy 2-layer, 4-head transformer on a repeated-token synthetic dataset, our pipeline surfaced an unexpected, non-trivial result:

| Measurement Approach | Primary Target Identified | Metric Value / Impact |
| :--- | :--- | :--- |
| **Methodology 1: Attention Score** | **Layer 0, Head 1 (L0H1)** | `0.91` (Perfect Shifted Diagonal) |
| **Methodology 2: Causal Ablation** | **Layer 1, Head 3 (L1H3)** | Highest Accuracy Drop ($\Delta \mathcal{L} = +2.41$) |

### The Analytical Paradox
* **L0H1** looks exactly where an induction head *should* look, yet ablating it causes negligible performance drops.
* **L1H3** does not showcase a pristine shifted-diagonal attention map, yet knocking it out completely breaks the model's capacity to complete the induction sequence.

---

## 💡 Deep Insights & Interpretability Conclusions

This divergence is **not a failure** of the experimental setup; rather, it is a textbook confirmation of how computation actually distributes itself across transformer circuits.
