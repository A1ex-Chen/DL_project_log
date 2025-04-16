def backup_model_best(self, filename, **kwargs):
    if not os.path.isabs(filename):
        filename = os.path.join(self.checkpoint_dir, filename)
    if os.path.exists(filename):
        backup_dir = os.path.join(self.checkpoint_dir, 'backup_model_best')
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        ts = datetime.datetime.now().timestamp()
        filename_backup = os.path.join(backup_dir, '%s.pt' % ts)
        shutil.copy(filename, filename_backup)
