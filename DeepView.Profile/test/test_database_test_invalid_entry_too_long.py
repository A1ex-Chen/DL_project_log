def test_invalid_entry_too_long(self):
    assert self.energy_table_interface.is_valid_entry([1, 2, 3, 4, 5]) is False
