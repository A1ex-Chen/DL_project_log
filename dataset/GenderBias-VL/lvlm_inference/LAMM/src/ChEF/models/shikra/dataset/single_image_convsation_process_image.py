def process_image(self, image: Image.Image) ->Dict[str, Any]:
    """
        convert Image.Image object to torch.Tensor
        """
    return self.process_func['image'](image, self.preprocessor)
