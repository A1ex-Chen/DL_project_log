def test_energy_table_is_created(self):
    query_result = self.test_database.connection.execute(
        "SELECT name from sqlite_schema WHERE type='table' and name ='ENERGY';"
        )
    query_result_list = query_result.fetchall()
    assert len(query_result_list) > 0
