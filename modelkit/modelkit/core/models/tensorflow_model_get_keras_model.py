def get_keras_model(self):


    class LayerFromSavedModel(tf.keras.layers.Layer):
        """
            Builds a keras Layer from a saved Tensorflow model
            """

        def __init__(self, saved_model_signature):
            super().__init__()
            self.saved_model_signature = saved_model_signature

        def call(self, inputs):
            return self.saved_model_signature(**{key: tf.convert_to_tensor(
                value) for key, value in inputs.items()})


    class KerasModelFromSavedModel(tf.keras.Model):
        """
            Builds a keras Model from a saved Tensorflow model
            via an artificial keras Layer built from the saved model
            signature.
            """

        def __init__(self, saved_model_signature):
            super().__init__()
            self.layer = LayerFromSavedModel(saved_model_signature)

        def call(self, inputs):
            out = self.layer(inputs)
            return out
    return KerasModelFromSavedModel(self.tf_model_signature)
