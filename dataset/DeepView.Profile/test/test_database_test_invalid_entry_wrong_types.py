def test_invalid_entry_wrong_types(self):
    assert self.energy_table_interface.is_valid_entry([None, None, None,
        None, None]) is False
