def get_object_name(self, name, version):
    return '/'.join((self.prefix, name, version))
