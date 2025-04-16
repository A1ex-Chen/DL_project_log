def allreduce_params():
    if not self.delay_allreduce:
        if self.needs_refresh:
            self.sync_bucket_structure()
            self.needs_refresh = False
    self.allreduce_fallback()
