def _compute_dependencies_load_info(self):
    global_load_info = {}
    add_dependencies_load_info(global_load_info, self)
    global_load_memory = (self._load_memory_increment or 0) + sum(x[
        'memory_increment'] for x in global_load_info.values())
    global_load_time = (self._load_time or 0) + sum(x['time'] for x in
        global_load_info.values())
    return global_load_time, global_load_memory
