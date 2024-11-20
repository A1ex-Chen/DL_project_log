def test_adding_valid_entry(self):
    params = ['entry_point', random.random(), random.random(), random.
        randint(LOWER_BOUND_RAND_INT, UPPER_BOUND_RAND_INT)]
    self.energy_table_interface.add_entry(params)
    query_result = self.test_database.connection.execute(
        'SELECT * FROM ENERGY ORDER BY ts DESC;').fetchone()
    assert query_result == tuple(params)
