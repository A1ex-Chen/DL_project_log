def exists(self, object_name):
    return os.path.isfile(os.path.join(self.bucket, *object_name.split('/')))
