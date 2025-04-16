def visualize_training(self, batched_inputs, results):
    from detectron2.utils.visualizer import Visualizer
    assert len(batched_inputs) == len(results
        ), 'Cannot visualize inputs and results of different sizes'
    storage = get_event_storage()
    max_boxes = 20
    image_index = 0
    img = batched_inputs[image_index]['image'].cpu().numpy()
    assert img.shape[0] == 3, 'Images should have 3 channels.'
    if self.input_format == 'BGR':
        img = img[::-1, :, :]
    img = img.transpose(1, 2, 0)
    v_gt = Visualizer(img, None)
    v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index][
        'instances'].gt_boxes)
    anno_img = v_gt.get_image()
    processed_results = detector_postprocess(results[image_index], img.
        shape[0], img.shape[1])
    predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy(
        )
    v_pred = Visualizer(img, None)
    v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
    prop_img = v_pred.get_image()
    vis_img = np.vstack((anno_img, prop_img))
    vis_img = vis_img.transpose(2, 0, 1)
    vis_name = (
        f'Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results')
    storage.put_image(vis_name, vis_img)
