def attn_hook(self, name, query, key, value, attention_probs):
    if name in DEFAULT_GUIDANCE_ATTN_KEYS:
        self._saved_attn[name] = attention_probs
