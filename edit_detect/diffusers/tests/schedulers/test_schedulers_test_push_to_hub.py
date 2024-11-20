def test_push_to_hub(self):
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', clip_sample=False, set_alpha_to_one=
        False)
    scheduler.push_to_hub(self.repo_id, token=TOKEN)
    scheduler_loaded = DDIMScheduler.from_pretrained(f'{USER}/{self.repo_id}')
    assert type(scheduler) == type(scheduler_loaded)
    delete_repo(token=TOKEN, repo_id=self.repo_id)
    with tempfile.TemporaryDirectory() as tmp_dir:
        scheduler.save_config(tmp_dir, repo_id=self.repo_id, push_to_hub=
            True, token=TOKEN)
    scheduler_loaded = DDIMScheduler.from_pretrained(f'{USER}/{self.repo_id}')
    assert type(scheduler) == type(scheduler_loaded)
    delete_repo(token=TOKEN, repo_id=self.repo_id)
