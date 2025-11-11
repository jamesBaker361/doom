from datasets import load_dataset, Dataset
import random

load_dataset("jlbaker361/sonic-vae-preprocessed-0.1",split="train").select(range(500)).push_to_hub("jlbaker361/sonic-vae-preprocessed-500")