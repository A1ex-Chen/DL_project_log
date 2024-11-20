def encode_file(self, path, ordered=False, verbose=False, add_eos=True,
    add_double_eos=False) ->torch.LongTensor:
    cached = path + '.bpe'
    if os.path.exists(cached):
        return torch.load(cached)
    print(f'encoding file {path} ...')
    assert os.path.exists(path), f"{path} doesn't exist"
    with open(path, encoding='utf-8') as f:
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stderr(
            devnull):
            out = torch.LongTensor(self.tokenizer.encode(f.read()) + [self.EOT]
                )
            with utils.distributed.sync_workers() as rank:
                if rank == 0:
                    torch.save(out, cached)
            return out
