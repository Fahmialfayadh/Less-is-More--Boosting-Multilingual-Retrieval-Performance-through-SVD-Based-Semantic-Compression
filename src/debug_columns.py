from datasets import load_dataset
ds = load_dataset('LazarusNLP/stsb_mt_id', split='test')
print("Columns:", ds.column_names)
print("First row:", ds[0])
