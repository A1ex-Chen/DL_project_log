def get_outputs(inputs, output_dir, suffix, output_format):
    fnames = (i[:-len('.json')].rsplit('/', maxsplit=1) for i in inputs)
    return [f'{output_dir or dirname}/{fname}{suffix}.{output_format}' for 
        dirname, fname in fnames]
