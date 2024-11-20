def log_grads_tb(tb_total_steps, grads, tb_subset='train'):
    tb_loggers[tb_subset].log_grads(tb_total_steps, grads)
