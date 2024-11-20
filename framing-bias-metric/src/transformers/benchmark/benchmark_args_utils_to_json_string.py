def to_json_string(self):
    """
        Serializes this instance to a JSON string.
        """
    return json.dumps(dataclasses.asdict(self), indent=2)
