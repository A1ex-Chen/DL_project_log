def visualize_training(self, batched_inputs, proposals):
    """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
    from detectron2.utils.visualizer import Visualizer
    storage = get_event_storage()
    max_vis_prop = 20
    for input, prop in zip(batched_inputs, proposals):
        img = input['image']
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=input['instances'].gt_boxes)
        anno_img = v_gt.get_image()
        box_size = min(len(prop.proposal_boxes), max_vis_prop)
        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=prop.proposal_boxes[0:
            box_size].tensor.cpu().numpy())
        prop_img = v_pred.get_image()
        vis_img = np.concatenate((anno_img, prop_img), axis=1)
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = 'Left: GT bounding boxes;  Right: Predicted proposals'
        storage.put_image(vis_name, vis_img)
        break
