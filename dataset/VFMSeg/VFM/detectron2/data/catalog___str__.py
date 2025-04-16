def __str__(self):
    return 'MetadataCatalog(registered metadata: {})'.format(', '.join(self
        .keys()))
