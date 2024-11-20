def delete(objId, sTime):
    """
			Delete rows from the temporary SQL table which are no longer required.
			This speeds up future queries.
			"""
    margin = 0
    cmd = 'DELETE FROM marker WHERE objectId = "{}" AND endTime < {}'.format(
        objId, sTime - margin)
    self.db.execute(cmd)
