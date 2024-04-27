import sys
import torch
import time
from hydragen_algo import hydragen_attention, attention_prefix

torch.set_default_device("cuda")
torch.manual_seed(1337)

b = 1024
nq = 1
prefix_len = 8192
suffix_len = 1
hq = 32
hkv = 8
d = 128

GENERATION_LEN = 512

# torch.cuda.memory._record_memory_history()

prefix_k = torch.randn(prefix_len, hkv, d, dtype=torch.bfloat16)
prefix_v = torch.randn(prefix_len, hkv, d, dtype=torch.bfloat16)
suffix_k = torch.randn(b, suffix_len, hkv, d, dtype=torch.bfloat16)
suffix_v = torch.randn(b, suffix_len, hkv, d, dtype=torch.bfloat16)

timing = []

# Benchmark decoding
s = time.time_ns()
for _ in range(GENERATION_LEN):
    q = torch.randn(b, nq, hq, d, dtype=torch.bfloat16)
    new_k = torch.randn(b, nq, hkv, d, dtype=torch.bfloat16)
    new_v = torch.randn(b, nq, hkv, d, dtype=torch.bfloat16)

    suffix_k = torch.cat((suffix_k, new_k), dim=1)
    suffix_v = torch.cat((suffix_v, new_v), dim=1)

    hydragen_out = hydragen_attention(q, prefix_k, prefix_v, suffix_k, suffix_v)

e = time.time_ns()

# # torch.cuda.memory._dump_snapshot("artifacts/hydragen_memory_dump.pickle")
# # torch.save(hydragen_out.to('cpu'), 'artifacts/hydragen_out.pt')
print(hydragen_out)
total_time = (e-s)/1_000_000_000
print(f"Total time: {total_time} seconds")
print(f"Tok/s: {(GENERATION_LEN*b)/total_time}")
print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9} GB")

# Benchmark Attention computation part only
# for _ in range(GENERATION_LEN):
#     print(_)

#     q = torch.randn(b, nq, hq, d, dtype=torch.bfloat16)
#     new_k = torch.randn(b, nq, hkv, d, dtype=torch.bfloat16)
#     new_v = torch.randn(b, nq, hkv, d, dtype=torch.bfloat16)

#     suffix_k = torch.cat((suffix_k, new_k), dim=1)
#     suffix_v = torch.cat((suffix_v, new_v), dim=1)

#     s = time.time_ns()
#     hydragen_out = hydragen_attention(q, prefix_k, prefix_v, suffix_k, suffix_v)
#     e = time.time_ns()

#     if _ >= 10: # Account for bit of warmup
#         timing.append((e-s)/1_000_000)
#     print("Hydragen out shape:", hydragen_out.shape)

# # torch.cuda.memory._dump_snapshot("artifacts/hydragen_memory_dump.pickle")
# # torch.save(hydragen_out.to('cpu'), 'artifacts/hydragen_out.pt')
# print(hydragen_out)
# print(f"Average time: {sum(timing)/len(timing)} ms")