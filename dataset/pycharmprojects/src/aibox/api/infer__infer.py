def _infer(task_name: Task.Name, path_to_checkpoint_or_checkpoint: Union[
    str, Checkpoint], lower_prob_thresh: float, upper_prob_thresh: float,
    device_ids: Optional[List[int]], path_to_image_list: List[str],
    path_to_results_dir: Optional[str]) ->Union[Tuple[Dict[str, List[str]],
    Dict[str, str], Dict[str, float]], Tuple[Dict[str, List[str]], Dict[str,
    List[BBox]], Dict[str, List[str]], Dict[str, List[float]]], Tuple[Dict[
    str, List[str]], Dict[str, List[BBox]], Dict[str, List[str]], Dict[str,
    List[float]], Dict[str, List[int]], Dict[str, List[int]], Dict[str,
    List[List[Tuple[int, int]]]], Dict[str, str]], Tuple[Dict[str, List[str
    ]], Dict[str, str], Dict[str, float]]]:
    if device_ids is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_count = 1 if not torch.cuda.is_available(
            ) else torch.cuda.device_count()
    else:
        device = torch.device('cuda', device_ids[0]) if len(device_ids
            ) > 0 else torch.device('cpu')
        device_count = len(device_ids) if len(device_ids) > 0 else 1
    if task_name == Task.Name.CLASSIFICATION:
        from aibox.lib.task.classification.checkpoint import Checkpoint
        from aibox.lib.task.classification.model import Model
        from aibox.lib.task.classification.preprocessor import Preprocessor
        from aibox.lib.task.classification.inferer import Inferer
    elif task_name == Task.Name.DETECTION:
        from aibox.lib.task.detection.checkpoint import Checkpoint
        from aibox.lib.task.detection.model import Model
        from aibox.lib.task.detection.preprocessor import Preprocessor
        from aibox.lib.task.detection.inferer import Inferer
    else:
        raise ValueError
    print('Preparing model...')
    time_checkpoint = time.time()
    if isinstance(path_to_checkpoint_or_checkpoint, str):
        path_to_checkpoint = path_to_checkpoint_or_checkpoint
        checkpoint = Checkpoint.load(path_to_checkpoint, device)
        model: Model = checkpoint.model
    elif isinstance(path_to_checkpoint_or_checkpoint, Checkpoint):
        checkpoint = path_to_checkpoint_or_checkpoint
        model: Model = checkpoint.model
    else:
        raise TypeError
    elapsed_time = time.time() - time_checkpoint
    print('Ready! Elapsed {:.2f} secs'.format(elapsed_time))
    batch_size = device_count
    preprocessor: Preprocessor = model.preprocessor
    inferer = Inferer(model, device_ids)
    print('Start inferring with {:s} (batch size: {:d})'.format('CPU' if 
        device == torch.device('cpu') else '{:d} GPUs'.format(device_count),
        batch_size))
    time_checkpoint = time.time()
    inference_list = []
    image_list = []
    process_dict_list = []
    for path_to_image in tqdm(path_to_image_list):
        image = Image.open(path_to_image).convert('RGB')
        processed_image, process_dict = preprocessor.process(image,
            is_train_or_eval=False)
        inference = inferer.infer(image_batch=[processed_image],
            lower_prob_thresh=lower_prob_thresh, upper_prob_thresh=
            upper_prob_thresh)
        inference_list.append(inference)
        image_list.append(image)
        process_dict_list.append(process_dict)
    elapsed_time = time.time() - time_checkpoint
    print('Done! Elapsed {:.2f} secs'.format(elapsed_time))
    if task_name == Task.Name.CLASSIFICATION:
        path_to_image_to_base64_images_dict = {}
        path_to_image_to_final_pred_category_dict = {}
        path_to_image_to_final_pred_prob_dict = {}
        for path_to_image, image, inference, process_dict in zip(
            path_to_image_list, image_list, inference_list, process_dict_list):
            grad_cam = inference.grad_cam_batch[0]
            final_pred_class = inference.final_pred_class_batch[0].item()
            final_pred_prob = inference.final_pred_prob_batch[0].item()
            final_pred_category = model.class_to_category_dict[final_pred_class
                ]
            print(f'Predicted category: {final_pred_category}')
            draw_images = []
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            text = f'{final_pred_category}: {final_pred_prob:.3f}'
            w, h = 6 * len(text), 10
            left, top = (draw_image.width - w) / 2, draw_image.height - 30
            draw.rectangle(((left, top), (left + w, top + h)), fill='gray')
            draw.text((left, top), text, fill='white')
            draw_images.append(draw_image)
            draw_image = image.copy()
            color_map = cm.get_cmap('jet')
            heatmap = color_map((grad_cam.cpu().numpy() * 255).astype(np.uint8)
                )
            heatmap = Preprocessor.inv_process_heatmap(process_dict, heatmap)
            heatmap[:, :, 3] = 0.5
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = Image.fromarray(heatmap).convert('RGBA')
            draw_image = draw_image.convert('RGBA')
            draw_image = Image.alpha_composite(draw_image, heatmap)
            draw_image = draw_image.convert('RGB')
            draw_images.append(draw_image)
            base64_images = []
            for draw_image in draw_images:
                buffer = BytesIO()
                draw_image.save(buffer, format='JPEG')
                base64_image = base64.b64encode(buffer.getvalue()).decode(
                    'utf-8')
                base64_images.append(base64_image)
            if path_to_results_dir is not None:
                os.makedirs(path_to_results_dir, exist_ok=True)
                filename = os.path.basename(path_to_image)
                path_to_output_image = os.path.join(path_to_results_dir,
                    filename)
                draw_images[0].save(path_to_output_image)
                stem, _ = os.path.splitext(filename)
                path_to_output_image = os.path.join(path_to_results_dir,
                    f'{stem}-heatmap.png')
                draw_images[1].save(path_to_output_image)
            path_to_image_to_base64_images_dict[path_to_image] = base64_images
            path_to_image_to_final_pred_category_dict[path_to_image
                ] = final_pred_category
            path_to_image_to_final_pred_prob_dict[path_to_image
                ] = final_pred_prob
        return (path_to_image_to_base64_images_dict,
            path_to_image_to_final_pred_category_dict,
            path_to_image_to_final_pred_prob_dict)
    elif task_name == Task.Name.DETECTION:
        path_to_image_to_base64_images_dict = {}
        path_to_image_to_final_detection_bboxes_dict = {}
        path_to_image_to_final_detection_categories_dict = {}
        path_to_image_to_final_detection_probs_dict = {}
        for path_to_image, image, inference, process_dict in zip(
            path_to_image_list, image_list, inference_list, process_dict_list):
            anchor_bboxes = inference.anchor_bboxes_batch[0]
            proposal_bboxes = inference.proposal_bboxes_batch[0]
            proposal_probs = inference.proposal_probs_batch[0]
            detection_bboxes = inference.detection_bboxes_batch[0]
            detection_classes = inference.detection_classes_batch[0]
            detection_probs = inference.detection_probs_batch[0]
            final_detection_bboxes = inference.final_detection_bboxes_batch[0]
            final_detection_classes = inference.final_detection_classes_batch[0
                ]
            final_detection_probs = inference.final_detection_probs_batch[0]
            anchor_bboxes = Preprocessor.inv_process_bboxes(process_dict,
                anchor_bboxes)
            proposal_bboxes = Preprocessor.inv_process_bboxes(process_dict,
                proposal_bboxes)
            detection_bboxes = Preprocessor.inv_process_bboxes(process_dict,
                detection_bboxes)
            final_detection_bboxes = Preprocessor.inv_process_bboxes(
                process_dict, final_detection_bboxes)
            anchor_bboxes = anchor_bboxes[BBox.inside(anchor_bboxes, left=0,
                top=0, right=image.width, bottom=image.height)].tolist()
            proposal_bboxes = proposal_bboxes.tolist()
            proposal_probs = proposal_probs.tolist()
            detection_bboxes = detection_bboxes.tolist()
            detection_categories = [model.class_to_category_dict[cls] for
                cls in detection_classes.tolist()]
            detection_probs = detection_probs.tolist()
            final_detection_bboxes = final_detection_bboxes.tolist()
            final_detection_categories = [model.class_to_category_dict[cls] for
                cls in final_detection_classes.tolist()]
            final_detection_probs = final_detection_probs.tolist()
            is_bright = ImageStat.Stat(image.convert('L')).rms[0] > 127
            offset = 0 if is_bright else 128
            category_to_color_dict = {category: tuple(random.randrange(0 +
                offset, 128 + offset) for _ in range(3)) for category in
                set(detection_categories)}
            draw_images = []
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            for bbox in anchor_bboxes:
                bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2],
                    bottom=bbox[3])
                draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.
                    bottom)), outline=(255, 0, 0))
            draw_images.append(draw_image)
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image, 'RGBA')
            min_proposal_probs, max_proposal_probs = min(proposal_probs), max(
                proposal_probs)
            for bbox, prob in zip(proposal_bboxes, proposal_probs):
                bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2],
                    bottom=bbox[3])
                alpha = int((prob - min_proposal_probs) / (
                    max_proposal_probs - min_proposal_probs) * 255)
                draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.
                    bottom)), outline=(255, 0, 0, alpha), width=2)
            draw_images.append(draw_image)
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image, 'RGBA')
            min_detection_probs, max_detection_probs = min(detection_probs
                ), max(detection_probs)
            for bbox, category, prob in zip(detection_bboxes,
                detection_categories, detection_probs):
                bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2],
                    bottom=bbox[3])
                color = category_to_color_dict[category]
                alpha = int((prob - min_detection_probs) / (
                    max_detection_probs - min_detection_probs) * 255)
                draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.
                    bottom)), outline=color + (alpha,), width=2)
            draw_images.append(draw_image)
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            for bbox, category, prob in zip(final_detection_bboxes,
                final_detection_categories, final_detection_probs):
                bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2],
                    bottom=bbox[3])
                color = category_to_color_dict[category]
                text = '[{:d}] {:s} {:.3f}'.format(model.
                    category_to_class_dict[category], category if category.
                    isascii() else '', prob)
                draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.
                    bottom)), outline=color, width=2)
                draw.rectangle(((bbox.left, bbox.top + 10), (bbox.left + 6 *
                    len(text), bbox.top)), fill=color)
                draw.text((bbox.left, bbox.top), text, fill='white' if
                    is_bright else 'black')
            draw_images.append(draw_image)
            base64_images = []
            for draw_image in draw_images:
                buffer = BytesIO()
                draw_image.save(buffer, format='JPEG')
                base64_image = base64.b64encode(buffer.getvalue()).decode(
                    'utf-8')
                base64_images.append(base64_image)
            if path_to_results_dir is not None:
                os.makedirs(path_to_results_dir, exist_ok=True)
                filename = os.path.basename(path_to_image)
                path_to_output_image = os.path.join(path_to_results_dir,
                    filename)
                draw_images[-1].save(path_to_output_image)
            path_to_image_to_base64_images_dict[path_to_image] = base64_images
            path_to_image_to_final_detection_bboxes_dict[path_to_image
                ] = final_detection_bboxes
            path_to_image_to_final_detection_categories_dict[path_to_image
                ] = final_detection_categories
            path_to_image_to_final_detection_probs_dict[path_to_image
                ] = final_detection_probs
        return (path_to_image_to_base64_images_dict,
            path_to_image_to_final_detection_bboxes_dict,
            path_to_image_to_final_detection_categories_dict,
            path_to_image_to_final_detection_probs_dict)
    else:
        raise ValueError
