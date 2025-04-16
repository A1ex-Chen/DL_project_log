def get_list_of_data_files(GP):
    import pilot2_datasets as p2
    reload(p2)
    print('Reading Data...')
    data_set = p2.data_sets[GP['set_sel']][0]
    print('Reading Data Files... %s->%s' % (GP['set_sel'], data_set))
    data_file = candle.fetch_file(
        'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot2/' +
        data_set + '.tar.gz', unpack=True, subdir='Pilot2')
    data_dir = os.path.join(os.path.dirname(data_file), data_set)
    data_files = glob.glob('%s/*.npz' % data_dir)
    fields = p2.gen_data_set_dict()
    return data_files, fields
