def expand_group_to_two_letter_codes(self, grp_name):
    return [self.get_two_letter_code(x) for x in self.constituents[grp_name]]
