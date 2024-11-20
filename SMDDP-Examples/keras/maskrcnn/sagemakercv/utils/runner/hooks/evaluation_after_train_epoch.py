def after_train_epoch(self, runner):
    if self.per_epoch:
        self.eval_running = True
        self.threads = []
        if dist_utils.MPI_rank() == 0:
            runner.logger.info('Running eval for epoch {}'.format(runner.epoch)
                )
        for i, data in enumerate(self.eval_dataset):
            prediction = runner.trainer(data, training=False)
            prediction = {i: j.numpy() for i, j in prediction.items()}
            self.threads.append(self.thread_pool.submit(evaluation.
                process_prediction, prediction))
        self.comm.Barrier()
