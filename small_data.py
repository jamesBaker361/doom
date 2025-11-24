from datasets import load_dataset,Dataset

for zone in ["EmeraldHillZone","HillTopZone"]:
    data=f"jlbaker361/discrete_{zone}.Act110000"
    dest=f"jlbaker361/discrete_{zone}.Act1100"
    src=load_dataset(data,split="train").to_dict()
    finished={
        key:value[:100] for key,value in src.items()
    }
    
    Dataset.from_dict(finished).push_to_hub(dest)
    