from datasets import load_dataset
import json

try:
    ds = load_dataset('mteb/MIRACLRetrieval', 'id', split='dev')
    cols = ds.column_names
    row = ds[0]
    out = {"cols": cols, "row": row}
except Exception as e:
    out = {"error": str(e)}

with open('/content/cols.json', 'w') as f:
    json.dump(out, f)
