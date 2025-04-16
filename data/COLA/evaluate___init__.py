def __init__(self, config_str, performance, pred_label_list):
    self_dict = copy.deepcopy(config_str)
    for key, value in performance.items():
        self_dict[key] = value
    self.summary = self_dict
    self.pred_label_list = pred_label_list
