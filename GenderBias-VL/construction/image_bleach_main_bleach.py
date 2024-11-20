def main_bleach(self):
    args = self.args
    detect_prompt = args.detect_prompt
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    for i, base_data in tqdm(enumerate(self.base_dataloader), total=len(
        self.base_dataloader)):
        base_names, base_imgs, base_nsfw_labels, img_paths = base_data
        base_imgs = base_imgs.to(self.device)
        detect_prompts = [detect_prompt] * base_imgs.shape[0]
        base_bboxes_batch, base_predicts_batch, base_phrases_batch = (self.
            predict_batch(self.grounding_model, base_imgs, detect_prompts,
            box_threshold, text_threshold, device=self.device))
        for j in range(base_imgs.shape[0]):
            base_img = base_imgs[j]
            img_path = img_paths[j]
            base_name = base_names[j]
            base_phrases = base_phrases_batch[j]
            base_predicts = base_predicts_batch[j]
            base_bboxes = base_bboxes_batch[j]
            sam_img = cv2.imread(img_path)
            sam_img = cv2.cvtColor(sam_img, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(sam_img)
            H, W = base_img.shape[1], base_img.shape[2]
            for i in range(base_bboxes.size(0)):
                base_bboxes[i] = base_bboxes[i] * torch.Tensor([W, H, W, H])
                base_bboxes[i][:2] -= base_bboxes[i][2:] / 2
                base_bboxes[i][2:] += base_bboxes[i][:2]
            base_bboxes = base_bboxes.cpu()
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
                base_bboxes, sam_img.shape[:2]).to(self.device)
            masks, _, _ = self.sam_predictor.predict_torch(point_coords=
                None, point_labels=None, boxes=transformed_boxes.to(self.
                device), multimask_output=False)
            plt.figure(figsize=(10, 10))
            plt.imshow(sam_img)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box, label in zip(base_bboxes, base_phrases):
                show_box(box.numpy(), plt.gca(), label)
            plt.axis('off')
            save_fig_name = img_path.replace(src_dir, os.path.join(self.
                sub_exp_dir, 'grounded_sam'))
            if not os.path.exists(os.path.dirname(save_fig_name)):
                os.makedirs(os.path.dirname(save_fig_name), exist_ok=True)
            plt.savefig(save_fig_name, bbox_inches='tight')
            mask = masks[0][0].cpu().numpy()
            mask_pil = Image.fromarray(mask)
            image_pil = Image.fromarray(sam_img)
            sam_img[mask] = 127
            new_img = Image.fromarray(sam_img)
            save_fig_name = img_path.replace(src_dir, os.path.join(self.
                sub_exp_dir, 'edit_images'))
            if not os.path.exists(os.path.dirname(save_fig_name)):
                os.makedirs(os.path.dirname(save_fig_name), exist_ok=True)
            new_img.save(save_fig_name)
