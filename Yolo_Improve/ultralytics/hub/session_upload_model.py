def upload_model(self, epoch: int, weights: str, is_best: bool=False, map:
    float=0.0, final: bool=False) ->None:
    """
        Upload a model checkpoint to Ultralytics HUB.

        Args:
            epoch (int): The current training epoch.
            weights (str): Path to the model weights file.
            is_best (bool): Indicates if the current model is the best one so far.
            map (float): Mean average precision of the model.
            final (bool): Indicates if the model is the final model after training.
        """
    if Path(weights).is_file():
        progress_total = Path(weights).stat().st_size if final else None
        self.request_queue(self.model.upload_model, epoch=epoch, weights=
            weights, is_best=is_best, map=map, final=final, retry=10,
            timeout=3600, thread=not final, progress_total=progress_total,
            stream_response=True)
    else:
        LOGGER.warning(
            f'{PREFIX}WARNING ⚠️ Model upload issue. Missing model {weights}.')
