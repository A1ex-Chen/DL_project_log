def _create_report_tables(self):
    cursor = self._connection.cursor()
    cursor.execute(queries.set_report_format_version.format(version=
        OperationRunTimeReportBuilder.Version))
    for creation_query in queries.create_report_tables.values():
        cursor.execute(creation_query)
    self._connection.commit()
