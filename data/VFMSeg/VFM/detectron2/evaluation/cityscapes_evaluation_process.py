def process(self, inputs, outputs):
    from cityscapesscripts.helpers.labels import trainId2label
    for input, output in zip(inputs, outputs):
        file_name = input['file_name']
        basename = os.path.splitext(os.path.basename(file_name))[0]
        pred_filename = os.path.join(self._temp_dir, basename + '_pred.png')
        output = output['sem_seg'].argmax(dim=0).to(self._cpu_device).numpy()
        pred = 255 * np.ones(output.shape, dtype=np.uint8)
        for train_id, label in trainId2label.items():
            if label.ignoreInEval:
                continue
            pred[output == train_id] = label.id
        Image.fromarray(pred).save(pred_filename)
