def run(tasks, **download_kwargs):
    if not os.path.isfile(json_spec['file_path']) or not os.path.isfile(
        'LICENSE.txt'):
        print('Downloading JSON metadata...')
        download_files([json_spec, license_specs['json']], **download_kwargs)
    print('Parsing JSON metadata...')
    with open(json_spec['file_path'], 'rb') as f:
        json_data = json.load(f, object_pairs_hook=OrderedDict)
    if 'stats' in tasks:
        print_statistics(json_data)
    specs = []
    if 'images' in tasks:
        specs += [item['image'] for item in json_data.values()] + [
            license_specs['images']]
    if 'thumbs' in tasks:
        specs += [item['thumbnail'] for item in json_data.values()] + [
            license_specs['thumbs']]
    if 'wilds' in tasks:
        specs += [item['in_the_wild'] for item in json_data.values()] + [
            license_specs['wilds']]
    if 'tfrecords' in tasks:
        specs += tfrecords_specs + [license_specs['tfrecords']]
    if len(specs):
        print('Downloading %d files...' % len(specs))
        np.random.shuffle(specs)
        download_files(specs, **download_kwargs)
    if 'align' in tasks:
        recreate_aligned_images(json_data, source_dir=download_kwargs[
            'source_dir'], rotate_level=not download_kwargs['no_rotation'],
            random_shift=download_kwargs['random_shift'], enable_padding=
            not download_kwargs['no_padding'], retry_crops=download_kwargs[
            'retry_crops'])
