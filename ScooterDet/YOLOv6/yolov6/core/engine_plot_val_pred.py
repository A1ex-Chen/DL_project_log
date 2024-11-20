def plot_val_pred(self, vis_outputs, vis_paths, vis_conf=0.3, vis_max_box_num=5
    ):
    self.vis_imgs_list = []
    for vis_output, vis_path in zip(vis_outputs, vis_paths):
        vis_output_array = vis_output.cpu().numpy()
        ori_img = cv2.imread(vis_path)
        for bbox_idx, vis_bbox in enumerate(vis_output_array):
            x_tl = int(vis_bbox[0])
            y_tl = int(vis_bbox[1])
            x_br = int(vis_bbox[2])
            y_br = int(vis_bbox[3])
            box_score = vis_bbox[4]
            cls_id = int(vis_bbox[5])
            if box_score < vis_conf or bbox_idx > vis_max_box_num:
                break
            cv2.rectangle(ori_img, (x_tl, y_tl), (x_br, y_br), tuple([int(x
                ) for x in self.color[cls_id]]), thickness=1)
            cv2.putText(ori_img,
                f"{self.data_dict['names'][cls_id]}: {box_score:.2f}", (
                x_tl, y_tl - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple([int
                (x) for x in self.color[cls_id]]), thickness=1)
        self.vis_imgs_list.append(torch.from_numpy(ori_img[:, :, ::-1].copy()))
