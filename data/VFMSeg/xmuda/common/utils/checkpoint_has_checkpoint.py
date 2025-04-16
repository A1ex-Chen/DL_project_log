def has_checkpoint(self):
    save_file = os.path.join(self.save_dir, 'last_checkpoint' + self.postfix)
    return os.path.exists(save_file)
