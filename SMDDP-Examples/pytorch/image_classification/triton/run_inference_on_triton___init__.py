def __init__(self, server_url: str, model_name: str, model_version: str, *,
    dataloader, verbose=False, resp_wait_s: Optional[float]=None,
    max_unresponded_reqs: Optional[int]=None):
    self._server_url = server_url
    self._model_name = model_name
    self._model_version = model_version
    self._dataloader = dataloader
    self._verbose = verbose
    self._response_wait_t = (self.DEFAULT_MAX_RESP_WAIT_S if resp_wait_s is
        None else resp_wait_s)
    self._max_unresp_reqs = (self.DEFAULT_MAX_UNRESP_REQS if 
        max_unresponded_reqs is None else max_unresponded_reqs)
    self._results = queue.Queue()
    self._processed_all = False
    self._errors = []
    self._num_waiting_for = 0
    self._sync = threading.Condition()
    self._req_thread = threading.Thread(target=self.req_loop, daemon=True)
