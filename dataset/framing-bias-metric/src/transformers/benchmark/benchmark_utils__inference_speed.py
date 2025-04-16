@abstractmethod
def _inference_speed(self, model_name: str, batch_size: int,
    sequence_length: int) ->float:
    pass
