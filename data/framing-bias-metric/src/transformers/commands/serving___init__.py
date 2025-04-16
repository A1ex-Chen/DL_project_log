def __init__(self, pipeline: Pipeline, host: str, port: int, workers: int):
    self._pipeline = pipeline
    self.host = host
    self.port = port
    self.workers = workers
    if not _serve_dependencies_installed:
        raise RuntimeError(
            'Using serve command requires FastAPI and unicorn. Please install transformers with [serving]: pip install "transformers[serving]".Or install FastAPI and unicorn separately.'
            )
    else:
        logger.info('Serving model over {}:{}'.format(host, port))
        self._app = FastAPI(routes=[APIRoute('/', self.model_info,
            response_model=ServeModelInfoResult, response_class=
            JSONResponse, methods=['GET']), APIRoute('/tokenize', self.
            tokenize, response_model=ServeTokenizeResult, response_class=
            JSONResponse, methods=['POST']), APIRoute('/detokenize', self.
            detokenize, response_model=ServeDeTokenizeResult,
            response_class=JSONResponse, methods=['POST']), APIRoute(
            '/forward', self.forward, response_model=ServeForwardResult,
            response_class=JSONResponse, methods=['POST'])], timeout=600)
