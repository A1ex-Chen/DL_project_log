def check_required_exists(self, gparam):
    """Functionality to verify that the required
        model parameters have been specified.
        """
    key_set = set(gparam.keys())
    intersect_set = key_set.intersection(self.required)
    diff_set = self.required.difference(intersect_set)
    if len(diff_set) > 0:
        raise Exception(
            'ERROR ! Required parameters are not specified.  These required parameters have not been initialized: '
             + str(sorted(diff_set)) + '... Exiting')
