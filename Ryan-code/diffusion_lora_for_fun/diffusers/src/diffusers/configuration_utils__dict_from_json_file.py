@classmethod
def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
    with open(json_file, 'r', encoding='utf-8') as reader:
        text = reader.read()
    return json.loads(text)
