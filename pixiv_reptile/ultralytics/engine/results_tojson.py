def tojson(self, normalize=False, decimals=5):
    """Converts detection results to JSON format."""
    import json
    return json.dumps(self.summary(normalize=normalize, decimals=decimals),
        indent=2)
