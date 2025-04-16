def setup(self):
    self.vision_model = FlaxCLIPVisionModule(self.config.vision_config)
    self.visual_projection = nn.Dense(self.config.projection_dim, use_bias=
        False, dtype=self.dtype)
    self.concept_embeds = self.param('concept_embeds', jax.nn.initializers.
        ones, (17, self.config.projection_dim))
    self.special_care_embeds = self.param('special_care_embeds', jax.nn.
        initializers.ones, (3, self.config.projection_dim))
    self.concept_embeds_weights = self.param('concept_embeds_weights', jax.
        nn.initializers.ones, (17,))
    self.special_care_embeds_weights = self.param('special_care_embeds_weights'
        , jax.nn.initializers.ones, (3,))
