def createMarkerTable(self):
    """
		Create a temporary table and index it to speed up repeated SQL quesries.
		The table is an INNER JOIN of CUPTI_ACTIVITY_KIND_MARKER with itself.
		"""
    cmd = (
        'CREATE TEMPORARY TABLE marker AS SELECT \t\t\t\t\ta._id_ as id, \t\t\t\t\ta.timestamp AS startTime, \t\t\t\t\tb.timestamp AS endTime, \t\t\t\t\tHEX(a.objectId) AS objectId, \t\t\t\t\ta.name AS name \t\t\t\t\tFROM {} AS a INNER JOIN {} AS b ON \t\t\t\t\ta.id = b.id and \t\t\t\t\ta.flags = 2 and b.flags = 4'
        .format(self.markerT, self.markerT))
    self.db.execute(cmd)
    self.db.execute('CREATE INDEX start_index ON marker (startTime)')
    self.db.execute('CREATE INDEX end_index ON marker (endTime)')
    self.db.execute('CREATE INDEX id_index ON marker (id)')
