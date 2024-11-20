def get_object_uri(self, object_name, sub_part=None):
    return 'azfs://' + '/'.join((self.bucket, object_name, *(sub_part or ''
        ).split('/')))
