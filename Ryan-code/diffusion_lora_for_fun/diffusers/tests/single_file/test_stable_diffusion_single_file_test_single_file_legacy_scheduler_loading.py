def test_single_file_legacy_scheduler_loading(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_filename = self.ckpt_path.split('/')[-1]
        local_ckpt_path = download_single_file_checkpoint(self.repo_id,
            ckpt_filename, tmpdir)
        local_original_config = download_original_config(self.
            original_config, tmpdir)
        pipe = self.pipeline_class.from_single_file(local_ckpt_path,
            original_config=local_original_config, cache_dir=tmpdir,
            local_files_only=True, scheduler_type='euler')
    assert isinstance(pipe.scheduler, EulerDiscreteScheduler)
