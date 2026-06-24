from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

tokens = ['Ø£ÙĪØ¶Ø§Ø¹', 'ëĤ¡', 'ä¸ºç©º', 'ä»ĸä»¬']

for t in tokens:
    try:
        # We can find the token id and decode it
        token_id = tokenizer.convert_tokens_to_ids(t)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            decoded = tokenizer.decode([token_id])
            print(f"Token: {t} -> Decoded: {decoded}")
        else:
            print(f"Token: {t} -> Unknown token ID")
    except Exception as e:
        print(f"Token: {t} -> Error: {e}")
