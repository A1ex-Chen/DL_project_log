@abstractmethod
def _inference_memory(self, model_name: str, batch_size: int,
    sequence_length: int) ->[Memory, Optional[MemorySummary]]:
    pass
