def __loadResources(self, image_height, image_width, batch_size):
    self.stream = cuda.Stream()
    for model_name, obj in self.models.items():
        self.engine[model_name].allocate_buffers(shape_dict=obj.
            get_shape_dict(batch_size, image_height, image_width), device=
            self.torch_device)
