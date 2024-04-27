# Hydragen: High-Throughput LLM Inference with Shared Prefixes

The [paper](https://arxiv.org/abs/2402.05099) shows an efficient inference technique for LLMs when there are shared prefixes. Hydragen computes attention over the shared prefix and unique suffixes separately. This decomposition enables efficient prefix attention by batching queries together across sequences, reducing redundant memory reads and enabling the use of hardware-friendly matrix multiplications.

This repository has the core implementation of the algorithm and a comparison with FlashAttention.

At higher batch sizes, Flash Attention has a very high memory utilization, but Hydragen is able to handle it quite easily. 

## Results

The numbers below are based on a simulation of what happens in the attention module during incremental decoding, i.e. when a new query token comes in and then attention is computed with the past k-v values. At every step, the kv sequence length increases by 1.

The experiments below are done on a single A100-SXM-80GB GPU.

### MHA

Batch Size - 64
Prefix Length - 1024
Output Length - 512
#heads Q - 32
#heads KV - 32
Head Dim - 128

| Metric           | Flash Attention | Hydragen Attention |
|------------------|-----------------|--------------------|
| Tokens/s         | 5760            | 54183              |
| Peak Memory (GB) | 2.4             | 0.82               |
|                  |                 |                    |

Batch Size - 64
Prefix Length - 2048
Output Length - 512
#heads Q - 32
#heads KV - 32
Head Dim - 128

| Metric           | Flash Attention | Hydragen Attention |
|------------------|-----------------|--------------------|
| Tokens/s         | 2843            | 52424              |
| Peak Memory (GB) | 4.0             | 0.84               |
|                  |                 |                    |

Batch Size - 256
Prefix Length - 2048
Output Length - 512
#heads Q - 32
#heads KV - 32
Head Dim - 128

| Metric           | Flash Attention | Hydragen Attention |
|------------------|-----------------|--------------------|
| Tokens/s         | 996             | 10365              |
| Peak Memory (GB) | 16              | 3.26               |
|                  |                 |                    |

Batch Size - 256
Prefix Length - 2048
Output Length - 1024
#heads Q - 32
#heads KV - 32
Head Dim - 128

| Metric           | Flash Attention | Hydragen Attention |
|------------------|-----------------|--------------------|
| Tokens/s         | 889             | 4678               |
| Peak Memory (GB) | 19.33           | 6.48               |
|                  |                 |                    |

Batch Size - 256
Prefix Length - 4096
Output Length - 1024
#heads Q - 32
#heads KV - 32
Head Dim - 128

| Metric           | Flash Attention | Hydragen Attention |
|------------------|-----------------|--------------------|
| Tokens/s         | 477             | 4680               |
| Peak Memory (GB) | 32.22           | 6.52               |
|                  |                 |                    |

### GQA

Batch Size - 512
Prefix Length - 2048
Output Length - 512
#heads Q - 32
#heads KV - 8
Head Dim - 128

| Metric           | Flash Attention | Hydragen Attention |
|------------------|-----------------|--------------------|
| Tokens/s         | 5358            | 66894              |
| Peak Memory (GB) | 8.06            | 1.63               |
|                  |                 |                    |

Batch Size - 512
Prefix Length - 4096
Output Length - 512
#heads Q - 32
#heads KV - 8
Head Dim - 128

| Metric           | Flash Attention | Hydragen Attention |
|------------------|-----------------|--------------------|
| Tokens/s         | 2796            | 65739              |
| Peak Memory (GB) | 14.51           | 1.64               |
|                  |                 |                    |

Batch Size - 1024
Prefix Length - 4096
Output Length - 512
#heads Q - 32
#heads KV - 8
Head Dim - 128

| Metric           | Flash Attention | Hydragen Attention |
|------------------|-----------------|--------------------|
| Tokens/s         | 2032            | 41175              |
| Peak Memory (GB) | 29.01           | 3.26               |
|                  |                 |                    |

Batch Size - 1024
Prefix Length - 8192
Output Length - 512
#heads Q - 32
#heads KV - 8
Head Dim - 128

| Metric           | Flash Attention | Hydragen Attention |
|------------------|-----------------|--------------------|
| Tokens/s         | 939             | 40716              |
| Peak Memory (GB) | 54.78           | 3.28               |
|                  |                 |                    |

Batch Size - 1024
Prefix Length - 16384
Output Length - 512
#heads Q - 32
#heads KV - 8
Head Dim - 128

| Metric           | Flash Attention | Hydragen Attention |
|------------------|-----------------|--------------------|
| Tokens/s         | N/A             | 40497              |
| Peak Memory (GB) | OOM             | 3.31               |
|                  |                 |                    |