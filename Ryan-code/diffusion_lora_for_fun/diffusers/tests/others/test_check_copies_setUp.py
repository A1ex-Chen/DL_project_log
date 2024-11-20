def setUp(self):
    self.diffusers_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(self.diffusers_dir, 'schedulers/'))
    check_copies.DIFFUSERS_PATH = self.diffusers_dir
    shutil.copy(os.path.join(git_repo_path,
        'src/diffusers/schedulers/scheduling_ddpm.py'), os.path.join(self.
        diffusers_dir, 'schedulers/scheduling_ddpm.py'))
