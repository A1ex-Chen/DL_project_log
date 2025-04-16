def main_filter(self):
    self.write_msg(self.log, 'Start filtering\n')
    args = self.args
    detect_prompt = args.detect_prompt
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    for i, (base_data, cf_data) in tqdm(enumerate(zip(self.base_dataloader,
        self.cf_dataloader)), total=len(self.base_dataloader)):
        base_names, base_imgs, base_nsfw_labels = base_data
        cf_names, cf_imgs, cf_nsfw_labels = cf_data
        base_imgs, cf_imgs = base_imgs.to(self.device), cf_imgs.to(self.device)
        detect_prompts = [detect_prompt] * base_imgs.shape[0]
        base_bboxes_batch, base_predicts_batch, base_phrases_batch = (self.
            predict_batch(self.grounding_model, base_imgs, detect_prompts,
            box_threshold, text_threshold, device=self.device))
        cf_bboxes_batch, cf_predicts_batch, cf_phrases_batch = (self.
            predict_batch(self.grounding_model, cf_imgs, detect_prompts,
            box_threshold, text_threshold, device=self.device))
        for j in range(base_imgs.shape[0]):
            base_name = base_names[j]
            cf_name = cf_names[j]
            base_phrases = base_phrases_batch[j]
            cf_phrases = cf_phrases_batch[j]
            base_predicts = base_predicts_batch[j]
            cf_predicts = cf_predicts_batch[j]
            base_bboxes = base_bboxes_batch[j]
            cf_bboxes = cf_bboxes_batch[j]
            if len(base_bboxes) > 1:
                self.write_msg(self.log, f'{base_name} has multi faces\n')
                self.save_bbox_imgs(base_imgs[j], base_bboxes, base_phrases,
                    self.multi_face_dir, base_name)
                self.write_msg(self.multi_face, f'{base_name}\n')
            elif len(base_bboxes) == 0:
                self.write_msg(self.log, f'{base_name} has no face\n')
                self.save_bbox_imgs(base_imgs[j], base_bboxes, base_phrases,
                    self.no_face_dir, base_name)
                self.write_msg(self.no_face, f'{base_name}\n')
            elif len(base_bboxes) != len(cf_bboxes):
                self.write_msg(self.log,
                    f'{base_name} has different number of faces\n')
                self.save_bbox_imgs(base_imgs[j], base_bboxes, base_phrases,
                    self.diff_box_dir, 'base_' + base_name)
                self.save_bbox_imgs(cf_imgs[j], cf_bboxes, cf_phrases, self
                    .diff_box_dir, 'cf_' + cf_name)
                self.write_msg(self.diff_box, f'{base_name}\n')
            elif base_nsfw_labels[j] or cf_nsfw_labels[j]:
                self.write_msg(self.log, f'{base_name} is nsfw\n')
                self.save_bbox_imgs(base_imgs[j], base_bboxes, base_phrases,
                    self.nsfw_image_dir, 'base_' + base_name)
                self.save_bbox_imgs(cf_imgs[j], cf_bboxes, cf_phrases, self
                    .nsfw_image_dir, 'cf_' + cf_name)
                self.write_msg(self.nsfw_image, f'{base_name}\n')
            else:
                self.write_msg(self.test_case, f'{base_name}\n')
    self.write_msg(self.log, 'End filtering\n')
