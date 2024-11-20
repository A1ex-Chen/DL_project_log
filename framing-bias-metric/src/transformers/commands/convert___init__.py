def __init__(self, model_type: str, tf_checkpoint: str, pytorch_dump_output:
    str, config: str, finetuning_task_name: str, *args):
    self._logger = logging.get_logger('transformers-cli/converting')
    self._logger.info('Loading model {}'.format(model_type))
    self._model_type = model_type
    self._tf_checkpoint = tf_checkpoint
    self._pytorch_dump_output = pytorch_dump_output
    self._config = config
    self._finetuning_task_name = finetuning_task_name
