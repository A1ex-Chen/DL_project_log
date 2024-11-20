@property
def config(self):
    if self.is_loaded:
        return self.vision_tower.config
    else:
        return self.cfg_only
