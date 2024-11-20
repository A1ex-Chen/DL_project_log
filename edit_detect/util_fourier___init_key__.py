def __init_key__(self, top_key: torch.Tensor, sub_key: torch.Tensor):
    if top_key is None or sub_key is None:
        raise TypeError('')
    if not self.__top_dict__.is_key_exist(key=top_key):
        self.__top_dict__[top_key] = TensorDict(max_size=self.
            __sub_dict_max_size__)
    if not self.__top_dict__[top_key].is_key_exist(key=sub_key):
        self.__top_dict__[top_key][sub_key] = {DirectRecorder.SEQ_KEY:
            Recorder(), DirectRecorder.RECONST_KEY: None}
