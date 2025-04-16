def __init__(self, out_json, distributed=True, class_add_1=True):
    self._out_json = out_json
    self.class_add_1 = class_add_1
    self._distributed = distributed
    self._cpu_device = torch.device('cpu')
    self._logger = logging.getLogger(__name__)
    self._predictions = []
    self.reset()
