import os
import random
import deepview_profile.db.database as database

LOWER_BOUND_RAND_INT = 1
UPPER_BOUND_RAND_INT = 10
class MockDatabaseInterface(database.DatabaseInterface):


class TestSkylineDatabase:
    test_database: MockDatabaseInterface = MockDatabaseInterface("test.sqlite")
    energy_table_interface: database.EnergyTableInterface = (
        database.EnergyTableInterface(test_database.connection)
    )

    # Test if energy table is created

    # try adding invalid entry and test if it is added




    # add 10 valid entries and get top 3