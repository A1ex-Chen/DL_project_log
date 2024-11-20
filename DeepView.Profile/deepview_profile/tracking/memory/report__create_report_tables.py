def _create_report_tables(self):
    cursor = self._connection.cursor()
    cursor.execute(queries.set_report_format_version.format(version=
        MemoryReportBuilder.Version))
    for creation_query in queries.create_report_tables.values():
        cursor.execute(creation_query)
    cursor.executemany(queries.add_entry_type, map(lambda entry: (entry.
        value, entry.name), queries.EntryType))
    self._connection.commit()
