def flush_log():
    dllogger.flush()
    for tbl in tb_loggers.values():
        if tbl.enabled:
            tbl.summary_writer.flush()
