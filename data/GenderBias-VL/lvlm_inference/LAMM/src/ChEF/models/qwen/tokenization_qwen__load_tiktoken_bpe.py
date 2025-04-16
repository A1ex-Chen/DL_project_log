def _load_tiktoken_bpe(tiktoken_bpe_file: str) ->Dict[bytes, int]:
    with open(tiktoken_bpe_file, 'rb') as f:
        contents = f.read()
    return {base64.b64decode(token): int(rank) for token, rank in (line.
        split() for line in contents.splitlines() if line)}
