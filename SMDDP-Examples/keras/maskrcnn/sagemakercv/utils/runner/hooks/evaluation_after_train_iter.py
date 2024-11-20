def after_train_iter(self, runner):
    if self.every_n_inner_iters(runner, 25) and self.eval_running:
        if self.threads_done():
            imgIds, box_predictions, mask_predictions = self.format_threads()
            imgIds_mpi_list = self.comm.gather(imgIds, root=0)
            box_predictions_mpi_list = self.comm.gather(box_predictions, root=0
                )
            mask_predictions_mpi_list = self.comm.gather(mask_predictions,
                root=0)
            if dist_utils.MPI_rank() == 0:
                self.evaluate_thread = self.thread_pool.submit(self.
                    evaluate, imgIds_mpi_list, box_predictions_mpi_list,
                    mask_predictions_mpi_list, runner.logger, runner.iter,
                    runner.tensorboard_dir)
            self.eval_running = False
            self.comm.Barrier()
