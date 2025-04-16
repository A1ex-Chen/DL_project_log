def __init__(self, model, save_dir='', *, save_to_disk=None, **checkpointables
    ):
    is_main_process = comm.is_main_process()
    super().__init__(model, save_dir, save_to_disk=is_main_process if 
        save_to_disk is None else save_to_disk, **checkpointables)
    self.path_manager = PathManager
