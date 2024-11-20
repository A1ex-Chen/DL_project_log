def iterate_objects(self, prefix: Optional[str]=None):
    for filename in glob.iglob(os.path.join(self.bucket, os.path.join('**',
        '*')), recursive=True):
        if os.path.isfile(filename):
            yield '/'.join(os.path.split(os.path.relpath(filename, self.
                bucket)))
