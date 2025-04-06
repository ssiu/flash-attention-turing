import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from flash_attn_turing import flash_attn_func
import argparse


q = torch.randn(2,3, dtype=torch.float16).to("cuda")
k = torch.randn(2,3, dtype=torch.float16).to("cuda")
v = torch.randn(2,3, dtype=torch.float16).to("cuda")
print("hi")
# a, b = flash_attn_func(q,k,v,1,1,1,1)

a = flash_attn_func(q,k,v,1,1,1,1)
print(type(a))
# print(b)

