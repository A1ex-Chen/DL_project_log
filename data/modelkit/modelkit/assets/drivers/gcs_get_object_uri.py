def get_object_uri(self, object_name, sub_part=None):
    return 'gs://' + '/'.join((self.bucket, object_name, *(sub_part or '').
        split('/')))
