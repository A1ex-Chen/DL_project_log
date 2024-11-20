@abstractmethod
def update(self, current_frame: Any, trackelts: list[Tracklet]) ->tuple[
    list[Tracklet], list[dict[str, Any]]]:
    pass
