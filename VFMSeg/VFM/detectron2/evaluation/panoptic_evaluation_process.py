def process(self, inputs, outputs):
    from panopticapi.utils import id2rgb
    for input, output in zip(inputs, outputs):
        panoptic_img, segments_info = output['panoptic_seg']
        panoptic_img = panoptic_img.cpu().numpy()
        if segments_info is None:
            label_divisor = self._metadata.label_divisor
            segments_info = []
            for panoptic_label in np.unique(panoptic_img):
                if panoptic_label == -1:
                    continue
                pred_class = panoptic_label // label_divisor
                isthing = (pred_class in self._metadata.
                    thing_dataset_id_to_contiguous_id.values())
                segments_info.append({'id': int(panoptic_label) + 1,
                    'category_id': int(pred_class), 'isthing': bool(isthing)})
            panoptic_img += 1
        file_name = os.path.basename(input['file_name'])
        file_name_png = os.path.splitext(file_name)[0] + '.png'
        with io.BytesIO() as out:
            Image.fromarray(id2rgb(panoptic_img)).save(out, format='PNG')
            segments_info = [self._convert_category_id(x) for x in
                segments_info]
            self._predictions.append({'image_id': input['image_id'],
                'file_name': file_name_png, 'png_string': out.getvalue(),
                'segments_info': segments_info})
