def fetch_test_set(test_set_url):
    import wget
    fname = wget.download(test_set_url, 'opus_test.txt')
    lns = Path(fname).open().readlines()
    src = lmap(str.strip, lns[::4])
    gold = lmap(str.strip, lns[1::4])
    mar_model = lmap(str.strip, lns[2::4])
    assert len(gold) == len(mar_model) == len(src
        ), f'Gold, marian and source lengths {len(gold)}, {len(mar_model)}, {len(src)} mismatched'
    os.remove(fname)
    return src, mar_model, gold
