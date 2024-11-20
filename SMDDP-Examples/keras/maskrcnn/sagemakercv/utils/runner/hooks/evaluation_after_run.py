def after_run(self, runner):
    if not self.per_epoch:
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
    if dist_utils.MPI_rank() == 0:
        runner.logger.info('Processing final eval')
    while not self.threads_done():
        sleep(1)
    imgIds, box_predictions, mask_predictions = self.format_threads()
    imgIds_mpi_list = self.comm.gather(imgIds, root=0)
    box_predictions_mpi_list = self.comm.gather(box_predictions, root=0)
    mask_predictions_mpi_list = self.comm.gather(mask_predictions, root=0)
    if dist_utils.MPI_rank() == 0:
        self.evaluate(imgIds_mpi_list, box_predictions_mpi_list,
            mask_predictions_mpi_list, runner.logger, runner.iter, runner.
            tensorboard_dir)
    self.eval_running = False
    self.comm.Barrier()
