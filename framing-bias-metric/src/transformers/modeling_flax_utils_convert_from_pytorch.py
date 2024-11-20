@staticmethod
@abstractmethod
def convert_from_pytorch(pt_state: Dict, config: PretrainedConfig) ->Dict:
    raise NotImplementedError()
