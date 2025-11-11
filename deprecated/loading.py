from datasets import load_dataset
import torch

data=load_dataset("jlbaker361/sonic_emerald_100000_trained",split="train")

for row in data:
    break

print(type(row["posterior_list"]))

t=torch.tensor(row["posterior_list"])
print(t.size())