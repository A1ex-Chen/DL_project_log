def _measure_speed(self, func) ->float:
    with self.args.strategy.scope():
        try:
            if self.args.is_tpu or self.args.use_xla:
                logger.info(
                    'Do inference on TPU. Running model 5 times to stabilize compilation'
                    )
                timeit.repeat(func, repeat=1, number=5)
            runtimes = timeit.repeat(func, repeat=self.args.repeat, number=10)
            return min(runtimes) / 10.0
        except ResourceExhaustedError as e:
            self.print_fn("Doesn't fit on GPU. {}".format(e))
