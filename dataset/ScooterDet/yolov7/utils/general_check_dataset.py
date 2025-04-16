def check_dataset(dict):
    val, s = dict.get('val'), dict.get('download')
    if val and len(val):
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else
            [val])]
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [
                str(x) for x in val if not x.exists()])
            if s and len(s):
                print('Downloading %s ...' % s)
                if s.startswith('http') and s.endswith('.zip'):
                    f = Path(s).name
                    torch.hub.download_url_to_file(s, f)
                    r = os.system('unzip -q %s -d ../ && rm %s' % (f, f))
                else:
                    r = os.system(s)
                print('Dataset autodownload %s\n' % ('success' if r == 0 else
                    'failure'))
            else:
                raise Exception('Dataset not found.')
