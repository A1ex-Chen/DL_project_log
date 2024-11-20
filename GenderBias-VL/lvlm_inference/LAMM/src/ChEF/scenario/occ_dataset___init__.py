def __init__(self, base_data_path, meta_file, ppl=False, option_content=
    False, option_map=None, generative=False, **kwargs):
    self.base_data_path = base_data_path
    json_path = meta_file
    self.data = json.load(open(json_path, 'rb'))
    self.ppl = ppl
    self.option_content = option_content
    self.map_type = None
    if option_map != None:
        self.map_type = option_map['type']
        self.map_id = option_map['ids']
        if self.map_type != 'unnatural':
            self.option_map = OPTION_MAP[self.map_type][option_map['ids']]
    self.generative = generative
