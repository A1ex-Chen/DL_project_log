def parse_from_common(self):
    """Functionality to parse options common
        for all benchmarks.
        This functionality is based on methods 'get_default_neon_parser' and
        'get_common_parser' which are defined previously(above). If the order changes
        or they are moved, the calling has to be updated.
        """
    parser = self.parser
    if self.framework != 'neon':
        parser = get_default_neon_parser(parser)
    parser = get_common_parser(parser)
    self.parser = parser
    self.conffile = os.path.join(self.file_path, self.default_model)
