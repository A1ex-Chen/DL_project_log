def _get_has_nsfw_concepts(self, features, params):
    has_nsfw_concepts = self.safety_checker(features, params)
    return has_nsfw_concepts
