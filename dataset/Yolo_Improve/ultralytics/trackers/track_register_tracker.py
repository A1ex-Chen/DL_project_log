def register_tracker(model: object, persist: bool) ->None:
    """
    Register tracking callbacks to the model for object tracking during prediction.

    Args:
        model (object): The model object to register tracking callbacks for.
        persist (bool): Whether to persist the trackers if they already exist.
    """
    model.add_callback('on_predict_start', partial(on_predict_start,
        persist=persist))
    model.add_callback('on_predict_postprocess_end', partial(
        on_predict_postprocess_end, persist=persist))
