def __init__(self, base_data_path, ppl=False, option_content=False,
    option_map=None, img_crp=False, text_crp=False, split='31', generative=
    False, data_c_path='data/datasets/ChEF/ScienceQA_C', **kwargs):
    self.base_data_path = base_data_path
    json_path = os.path.join(self.base_data_path, 'meta_file',
        f'{self.task_name}_{self.dataset_name}.json')
    if text_crp:
        json_path = os.path.join(data_c_path, 'VQA_ScienceQA_C.json')
    self.data = json.load(open(json_path, 'rb'))
    self.ppl = ppl
    self.option_content = option_content
    self.map_type = None
    if option_map != None:
        self.map_type = option_map['type']
        self.map_id = option_map['ids']
        if self.map_type != 'unnatural':
            self.option_map = OPTION_MAP[self.map_type][option_map['ids']]
    self.data_c_path = data_c_path
    if img_crp:
        self.base_data_path = self.data_c_path
    self.img_crp = img_crp
    self.generative = generative
