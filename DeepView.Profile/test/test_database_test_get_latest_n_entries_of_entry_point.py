def test_get_latest_n_entries_of_entry_point(self):
    for _ in range(10):
        params = ['entry_point', random.random(), random.random(), random.
            randint(LOWER_BOUND_RAND_INT, UPPER_BOUND_RAND_INT)]
        self.energy_table_interface.add_entry(params)
    for _ in range(20):
        params = ['other_entry_point', random.random(), random.random(),
            random.randint(LOWER_BOUND_RAND_INT, UPPER_BOUND_RAND_INT)]
        self.energy_table_interface.add_entry(params)
    entries = []
    for _ in range(3):
        params = ['entry_point', random.random(), random.random(), random.
            randint(LOWER_BOUND_RAND_INT, UPPER_BOUND_RAND_INT)]
        entries.insert(0, params)
        self.energy_table_interface.add_entry(params)
    latest_n_entries = (self.energy_table_interface.
        get_latest_n_entries_of_entry_point(3, 'entry_point'))
    entries = [tuple(entry) for entry in entries]
    assert entries == latest_n_entries
