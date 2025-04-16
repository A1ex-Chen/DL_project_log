def to_labelbox_data(self: BBoxDataFrame, data_row: object, schema_lookup: dict
    ) ->list:
    """Convert a dataframe to the Labelbox format.

        Args:
            self (BBoxDataFrame): BBoxDataFrame object.
            data_row (DataRow): DataRow object.
            schema_lookup(dict): Dictionary of label names and label ids.

        Returns:
            uploads(list): List of dictionaries in Labelbox format.

        """
    segment = self.to_labelbox_segment()
    uploads = []
    for schema_name, schema_id in schema_lookup.items():
        if schema_name in segment:
            uploads.append({'uuid': str(uuid.uuid4()), 'schemaId':
                schema_id, 'dataRow': {'id': data_row.uid}, 'segments':
                segment[schema_name]})
    return uploads
