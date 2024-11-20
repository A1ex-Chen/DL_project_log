def find_recovery(self):
    recovery_path = os.path.join(self.recovery_dir, self.recovery_prefix)
    files = glob.glob(recovery_path + '*' + self.extension)
    files = sorted(files)
    return files[0] if len(files) else ''
