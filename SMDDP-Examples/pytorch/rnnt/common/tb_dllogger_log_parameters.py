def log_parameters(data, verbosity=0, tb_subset=None):
    for k, v in data.items():
        dllogger.log(step='PARAMETER', data={k: v}, verbosity=verbosity)
    if tb_subset is not None and tb_loggers[tb_subset].enabled:
        tb_data = {k: v for k, v in data.items() if type(v) in (str, bool,
            int, float)}
        tb_loggers[tb_subset].summary_writer.add_hparams(tb_data, {})
