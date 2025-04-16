def _measure_speed(self, func) ->float:
    try:
        if self.args.is_tpu or self.args.torchscript:
            logger.info(
                'Do inference on TPU or torchscript. Running model 5 times to stabilize compilation'
                )
            timeit.repeat(func, repeat=1, number=5)
        runtimes = timeit.repeat(func, repeat=self.args.repeat, number=10)
        if self.args.is_tpu and self.args.torch_xla_tpu_print_metrics:
            import torch_xla.debug.metrics as met
            self.print_fn(met.metrics_report())
        return min(runtimes) / 10.0
    except RuntimeError as e:
        self.print_fn("Doesn't fit on GPU. {}".format(e))
        return 'N/A'
