def __enter__(self):
    os.chdir(self.dir)
