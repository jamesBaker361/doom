from datasets import load_dataset

data=load_dataset("jlbaker361/sonic_emerald_100000",split="train")

n=100

small_dataset = data.shuffle(seed=42).select(range(n))

small_dataset.push_to_hub("jlbaker361/sonic_100")