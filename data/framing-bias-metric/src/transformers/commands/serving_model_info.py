def model_info(self):
    return ServeModelInfoResult(infos=vars(self._pipeline.model.config))
