def before_run(self, runner):
    self.eval_running = False
    self.thread_pool = ThreadPoolExecutor(max_workers=mp.cpu_count() //
        dist_utils.MPI_local_size())
    self.threads = []
