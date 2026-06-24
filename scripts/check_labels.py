from datasets import load_dataset
ds = load_dataset("mteb/NusaX-senti", "ind", split="train[:10]")
print(ds.features['label'])
print("Texts:", ds['text'][:3])
print("Labels:", ds['label'][:3])
