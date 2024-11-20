def fake_fetch_asset(asset_spec, return_info=True):
    nonlocal fetched
    fetched += 1
    return {'path': os.path.join(working_dir, 'something.txt')}
