def to_json_file(self, json_file_path):
    """ Save this instance to a json file."""
    with open(json_file_path, 'w', encoding='utf-8') as writer:
        writer.write(self.to_json_string())
