# Reverse Engineering Attention Heads for Induction Behaviour detection in Small Transformers

[![Research Area](https://img.shields.io/badge/Research-Mechanistic%20Interpretability-blueviolet)](https://github.com/priyamdas333/Mechanistic_induction_Interpretability)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Experimental%20Phase-success)](#)

This repository contains a mechanistic interpretability pipeline designed to open the "black box" of small, autoregressive transformer models. Specifically, we isolate, visualize, and causally validate **Induction Heads**—the fundamental circuits responsible for in-context learning, prefix-matching, and macro algorithmic pattern recognition in large language models.

---

## 🔬 Core Research Framework

First of all what is an induction behaviour?

It is an in-context learning behavior displayed by the attention heads and it is a 2 step mechanism involving at  least 2 heads. In simpler terms, one head is responsible for storing the current token and its previous one and the next head is responsible for pattern matching in order to predict the next token given the current one.

1.Previous token head is responsible for encoding the representation of current token and the previous token
2.Induction head performs pattern recognition and copying. As soon as it matches a repeating token,it searches the information stored by the previous token head as the token preceding the current one and use that previous token for predicting next token.

Mechanistically, the induction head breaks down into two specific circuits:

     •	The QK (Query-Key) Circuit (Where to attend): This determines where the model should focus its attention. For an induction head, the QK circuit learns to attend back to the previous occurrence of the current token.
     •	The OV (Output-Value) Circuit (What to copy): Once the attention is directed to the past occurrence, the OV circuit takes over. Its job is to take the token that followed that past occurrence and boost its logit (its probability score), effectively copying it as the prediction for the next token.
     
A Concrete Example: Imagine a sequence where the pattern "abcde" repeats. When the model is at the second "a" in "abcdeabcde", the induction head searches the past context, finds the first "a", notes that it was followed by "b", and uses that information to predict that "b" should come next.
Different circuit activations -> Triggering different attention heads -> Induction behaviour
Induction head matches current token with the previous token embedding held by the “previous token” head and hence can predict the next token

# Mechanistic Induction Interpretability

A mechanistic interpretability project investigating how induction behavior emerges inside small Transformer models.

Rather than treating transformers as black boxes, this project attempts to identify and understand the attention heads responsible for repeated-pattern recognition and in-context sequence completion.

The central question:

> When a Transformer learns to continue repeated patterns such as `[A][B]...[A] → [B]`, which attention heads are actually responsible?

---

# Motivation

Induction heads are among the most important known circuits discovered inside transformers.

They implement a simple but powerful algorithm:

```text
[A][B] ... [A]
          ↓
predict [B]
```

This match-and-copy behavior is believed to play a fundamental role in in-context learning and pattern completion.

Recent mechanistic interpretability research suggests that induction behavior often emerges through interactions between multiple attention heads rather than a single specialized component.

This project investigates whether induction heads can be identified through:

1. Attention pattern analysis
2. Causal ablation analysis

and whether these methods agree.

---

# Repository Goals

* Train a small Transformer on synthetic repeated-pattern datasets.
* Identify candidate induction heads.
* Compare correlation-based and causal identification methods.
* Analyze disagreements between the methods.
* Investigate whether induction behavior is distributed across multiple heads.

---

# Experimental Setup

## Model

* Decoder-only Transformer
* 2 Layers
* 4 Attention Heads per Layer

## Dataset

Synthetic repeated-pattern sequences.

Example:

```text
A B C D A B C
```

The model must learn:

```text
[A][B]...[A]
      ↓
predict [B]
```

which is the canonical induction task.

---

# Experiment 1: Attention-Based Induction Head Detection

## Hypothesis

An induction head should attend to locations corresponding to previous occurrences of the same pattern.

For every attention head, we compute an induction score:

```python
induction_score =
mean(attn[i, i-pattern_len])
```

which measures how strongly a head attends to the position associated with a previous pattern occurrence.

### Intuition

Consider:

```text
A B C D A B C
```

When processing the second occurrence of:

```text
A
```

an induction head should attend near the first occurrence.

This creates a characteristic shifted-diagonal structure in the attention matrix.

---

## Observation

The highest induction score was observed for:

```text
Layer 0 Head 1
```

This head consistently exhibited induction-like attention patterns.

---

## Interpretation

This suggests:

```text
Layer 0 Head 1
```

has learned to identify previous pattern occurrences.

The attention geometry strongly resembles the expected behavior of an induction-related head.

However:

> Attention reveals where information is retrieved from, not whether the retrieved information is actually important.

A head may attend to the correct location while contributing little to the final prediction.

---
# OV Circuit Analysis

There are 2 circuits associated with each attention_head:
1. QK circuit:WHERE to attend
2. OV circuit:WHAT to copy from the attended position

OVfull = WU ⋅ WO ⋅ WV ⋅ WE.T

OVfull[i,j] : If the attention head is attending to position j, how much it boosts the 
logit for predicting i

For an induction head: attending to token A should boost the logit of the token 
that follows A in the pattern.

This OV circuit analysis is very important in analyzing whether a particular attention head is actually following 
induction behaviour by boosting the logits of the previous spcific repeated sequence token.

# Experiment 2: Causal Head Ablation

## Hypothesis

If a head is functionally important for induction behavior, removing it should reduce pattern-completion performance.

For each attention head:

```python
head_output = 0
```

during inference.

We then measure the resulting drop in pattern-detection accuracy.

---

## Observation

The largest performance degradation occurred for:

```text
Layer 1 Head 3
```

---

## Interpretation

Unlike attention visualization, ablation measures causal importance.

This experiment asks:

> How much does the model actually rely on this head?

The result suggests:

```text
Layer 1 Head 3
```

plays a critical role in successful pattern completion.

---

# The Most Interesting Finding

The two methods did not identify the same head.

```text
Attention Analysis:
Layer 0 Head 1

Ablation Analysis:
Layer 1 Head 3
```

At first glance this appears contradictory.

However, it may indicate something more interesting:

```text
Layer 0 Head 1
        ↓
Layer 1 Head 3
```

a multi-head induction circuit.

---

# A Possible Circuit Hypothesis

One explanation is:

## Layer 0 Head 1

Learns:

```text
Find previous occurrence
of a pattern.
```

and writes information about pattern locations into the residual stream.

## Layer 1 Head 3

Learns:

```text
Use this information
to perform prediction.
```

and therefore becomes more important for final task accuracy.

Under this interpretation:

```text
L0H1 = pattern discovery

L1H3 = pattern utilization
```

Both heads are important.

They simply perform different roles.

---

## 📊 Empirical Observations & Discrepancies

| Measurement Approach | Primary Target Identified | Metric Value / Impact |
| :--- | :--- | :--- |
| **Methodology 1: Attention Score** | **Layer 0, Head 1 (L0H1)** | `0.91` (Perfect Shifted Diagonal) |
| **Methodology 2: Causal Ablation** | **Layer 1, Head 3 (L1H3)** | Highest Accuracy Drop ($\Delta \mathcal{L} = +2.41$) |

---
# Why This Matters

Many beginner interpretability projects assume:

```text
Highest attention score
=
Most important head
```

Our results suggest otherwise.

The head with the clearest induction-like attention pattern was not the head whose removal caused the largest performance degradation.

This indicates that:

* Attention patterns alone are insufficient for identifying functional importance.
* Causal interventions reveal dependencies that attention visualization cannot.
* Induction behavior may be distributed across multiple heads.

---

# Research Conclusions

### Observation 1

Attention-based induction scores successfully identify heads exhibiting induction-like behavior.

### Observation 2

Head ablation identifies heads that are causally important for prediction.

### Observation 3

The two methods may select different heads.

### Observation 4

This disagreement can reveal underlying circuits rather than isolated mechanisms.

### Observation 5

The strongest candidate explanation is that induction behavior emerges from cooperation between multiple heads across layers.

---

# Future Work

## Circuit Verification

Test whether:

```text
Layer 0 Head 1
        →
Layer 1 Head 3
```

forms a genuine induction circuit.

Possible experiment:

```python
ablate(L0H1)
```

and measure:

* accuracy drop
* attention changes in L1H3
* activation changes in L1H3

---

## Activation Patching

Replace activations from clean runs into corrupted runs and identify causal pathways.

---

## Attention Composition Analysis

Measure whether:

```text
L0H1 → L1H3
```

forms a compositional attention circuit.

---

## Residual Stream Analysis

Determine what information each head writes into the residual stream.

---

## Scaling Study

Investigate whether the same circuit structure appears in:

* larger toy transformers
* GPT-style models
* pretrained language models

---

# Key Takeaway

The most important result of this project is not simply identifying an induction head.

It is the discovery that:

```text
The head that looks most like an induction head
is not necessarily the head that matters most.
```

This suggests that induction behavior may emerge from interactions between multiple attention heads, providing evidence for circuit-level computation rather than single-head specialization.

Understanding these circuits is a step toward reverse-engineering how transformers perform in-context learning.

