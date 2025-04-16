def save_calib_cache_file(cache_file, activation_map, headline=
    'TRT-8XXX-EntropyCalibration2\n'):
    with open(os.path.join(cache_file), 'w') as cfile:
        cfile.write(headline)
        for k, v in activation_map.items():
            cfile.write('{}: {}\n'.format(k, v))
