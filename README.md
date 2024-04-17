# Hydragen: High-Throughput LLM Inference with Shared Prefixes

The [paper](https://arxiv.org/abs/2402.05099) shows an efficient inference technique for LLMs when there are shared prefixes. Hydragen computes attention over the shared prefix and unique suffixes separately. This decomposition enables efficient prefix attention by batching queries together across sequences, reducing redundant memory reads and enabling the use of hardware-friendly matrix multiplications.

This repository has the core implementation of the algortihm and a comparison with using FlashAttention.

At higher batch sizes, Flash Attention has a high mwmory utilzation, but Hydragen is able to handle it quite easily. 

Still a work in progress.