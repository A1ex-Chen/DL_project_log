def test_invalid_entry_too_short(self):
    assert self.energy_table_interface.is_valid_entry([]) is False
