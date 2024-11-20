@property
def gpu_options(self):
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    return options
