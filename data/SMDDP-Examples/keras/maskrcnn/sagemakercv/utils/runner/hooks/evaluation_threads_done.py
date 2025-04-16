def threads_done(self):
    local_done = all([i.done() for i in self.threads])
    all_done = self.comm.allgather(local_done)
    return all(all_done)
