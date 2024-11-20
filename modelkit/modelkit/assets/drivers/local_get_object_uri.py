def get_object_uri(self, object_name, sub_part=None):
    return os.path.join(self.bucket, *object_name.split('/'), *(sub_part.
        split('/') if sub_part else ()))
