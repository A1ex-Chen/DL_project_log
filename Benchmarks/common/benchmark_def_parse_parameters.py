def parse_parameters(self):
    """Functionality to parse options common
        for all benchmarks.
        This functionality is based on methods 'get_default_neon_parser' and
        'get_common_parser' which are defined previously(above). If the order changes
        or they are moved, the calling has to be updated.
        """
    self.parser = parsing_utils.parse_common(self.parser)
    self.parser = parsing_utils.parse_from_dictlist(self.
        additional_definitions, self.parser)
    self.conffile = os.path.join(self.file_path, self.default_model)
