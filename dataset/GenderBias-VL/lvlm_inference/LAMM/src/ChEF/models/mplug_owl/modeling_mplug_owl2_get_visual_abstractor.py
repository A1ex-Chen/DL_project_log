def get_visual_abstractor(self):
    visual_abstractor = getattr(self, 'visual_abstractor', None)
    if type(visual_abstractor) is list:
        visual_abstractor = visual_abstractor[0]
    return visual_abstractor
