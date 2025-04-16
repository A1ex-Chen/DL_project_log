def create_energy_table(self) ->None:
    self.connection.cursor().execute(
        'CREATE TABLE IF NOT EXISTS ENERGY (             entry_point TEXT,             cpu_component REAL,             gpu_component REAL,             batch_size INT,             ts TIMESTAMP         );'
        )
