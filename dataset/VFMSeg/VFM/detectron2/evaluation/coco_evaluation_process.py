def process(self, inputs, outputs):
    """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
    for input, output in zip(inputs, outputs):
        prediction = {'image_id': input['image_id']}
        if 'instances' in output:
            instances = output['instances'].to(self._cpu_device)
            prediction['instances'] = instances_to_coco_json(instances,
                input['image_id'])
        if 'proposals' in output:
            prediction['proposals'] = output['proposals'].to(self._cpu_device)
        if len(prediction) > 1:
            self._predictions.append(prediction)
