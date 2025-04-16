def encode_images(self, images):
    image_features = self.get_model().vision_model(images).last_hidden_state
    image_features = self.get_model().visual_abstractor(encoder_hidden_states
        =image_features).last_hidden_state
    return image_features
