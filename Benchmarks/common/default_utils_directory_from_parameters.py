def directory_from_parameters(params, commonroot='Output'):
    """Construct output directory path with unique IDs from parameters

    Parameters
    ----------
    params : python dictionary
        Dictionary of parameters read
    commonroot : string
        String to specify the common folder to store results.

    """
    if commonroot in set(['.', './']):
        outdir = os.path.abspath('.')
    else:
        outdir = os.path.abspath(os.path.join('.', commonroot))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outdir = os.path.abspath(os.path.join(outdir, params['experiment_id']))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outdir = os.path.abspath(os.path.join(outdir, params['run_id']))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    return outdir
