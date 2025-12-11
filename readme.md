# Sigmoid Attention for JAX/Flax

>  - Experimental **multi-head Sigmoid-based attention** module
>  - The test created as a small project for a medical image training model
>  - (JAX + Flax implementation, with `-log(N)` bias initialization trick)

---

## Overview

This repository provides a JAX/Flax implementation of a Transformer-style self-attention block where the usual softmax is replaced by a **sigmoid gate**.

Classic dot-product attention uses:

\[
\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)
\]

In this module, we use:

\[
\sigma\left(\frac{QK^\top}{\sqrt{d}} + b\right)
\]

with a bias initialized as:

\[
b \approx -\log N
\]

where \(N\) is the sequence length.  
This initialization makes the **expected sum of gates per query ≈ 1** at the beginning of training, so the overall scale is roughly comparable to softmax attention.

---

## Features

- **Multi-Head Sigmoid Attention**
  - Configurable `dim`, `num_hds`, `head_dim`
- **Bias initialization with `-log(N)`**
  - Bias is set based on sequence length `N`
  - Roughly one “effective” token per query at initialization (gate sum ≈ 1)
- **Optional Q/K LayerNorm**
  - `use_qk_norm=True` applies `LayerNorm` to `q` and `k` before the dot-product
- **Optional LayerScale**
  - `use_layerscale=True` multiplies the output by a learnable scale `γ`
- **Causal mask support**
  - `make_causal_mask(N)` builds a lower-triangular mask for autoregressive setups
- **JAX + Flax based**
  - Implemented as an `nn.Module` with `@nn.compact`
  - Ready to wrap with `jit` for fast experiments

---

## Todo


- **Bias initialization with `-log(N)`**
  - Bias will be dynamic based on each sequence length
- **After test on ViT tuned model**
  - `make_causal_mask(N)` will be corrected according to the changes with real-world dataset

---

## Dependency

- Python 3.10+
- [JAX](https://github.com/google/jax)
- [Flax](https://github.com/google/flax)

## Reference

- [Theory, Analysis, and Best Practices for Sigmoid Self-Attention](https://arxiv.org/abs/2409.04431), Authors: Jason Ramapuram et al. (Apple)
