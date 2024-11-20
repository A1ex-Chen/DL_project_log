def get_vision_tower(self):
    vision_model = getattr(self, 'vision_model', None)
    if type(vision_model) is list:
        vision_model = vision_model[0]
    return vision_model
