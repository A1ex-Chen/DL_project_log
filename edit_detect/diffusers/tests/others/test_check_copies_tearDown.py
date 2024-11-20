def tearDown(self):
    check_copies.DIFFUSERS_PATH = 'src/diffusers'
    shutil.rmtree(self.diffusers_dir)
