def generate_heatmap(self, im0, tracks):
    """
        Generate heatmap based on tracking data.

        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
        """
    self.im0 = im0
    if not self.initialized:
        self.heatmap = np.zeros((int(self.im0.shape[0]), int(self.im0.shape
            [1])), dtype=np.float32)
        self.initialized = True
    self.heatmap *= self.decay_factor
    self.extract_results(tracks)
    self.annotator = Annotator(self.im0, self.tf, None)
    if self.track_ids:
        if self.count_reg_pts is not None:
            self.annotator.draw_region(reg_pts=self.count_reg_pts, color=
                self.region_color, thickness=self.region_thickness)
        for box, cls, track_id in zip(self.boxes, self.clss, self.track_ids):
            if self.names[cls] not in self.class_wise_count:
                self.class_wise_count[self.names[cls]] = {'IN': 0, 'OUT': 0}
            if self.shape == 'circle':
                center = int((box[0] + box[2]) // 2), int((box[1] + box[3]) //
                    2)
                radius = min(int(box[2]) - int(box[0]), int(box[3]) - int(
                    box[1])) // 2
                y, x = np.ogrid[0:self.heatmap.shape[0], 0:self.heatmap.
                    shape[1]]
                mask = (x - center[0]) ** 2 + (y - center[1]
                    ) ** 2 <= radius ** 2
                self.heatmap[int(box[1]):int(box[3]), int(box[0]):int(box[2])
                    ] += 2 * mask[int(box[1]):int(box[3]), int(box[0]):int(
                    box[2])]
            else:
                self.heatmap[int(box[1]):int(box[3]), int(box[0]):int(box[2])
                    ] += 2
            track_line = self.track_history[track_id]
            track_line.append((float((box[0] + box[2]) / 2), float((box[1] +
                box[3]) / 2)))
            if len(track_line) > 30:
                track_line.pop(0)
            prev_position = self.track_history[track_id][-2] if len(self.
                track_history[track_id]) > 1 else None
            if self.count_reg_pts is not None:
                if len(self.count_reg_pts) >= 3:
                    is_inside = self.counting_region.contains(Point(
                        track_line[-1]))
                    if (prev_position is not None and is_inside and 
                        track_id not in self.count_ids):
                        self.count_ids.append(track_id)
                        if (box[0] - prev_position[0]) * (self.
                            counting_region.centroid.x - prev_position[0]) > 0:
                            self.in_counts += 1
                            self.class_wise_count[self.names[cls]]['IN'] += 1
                        else:
                            self.out_counts += 1
                            self.class_wise_count[self.names[cls]]['OUT'] += 1
                elif len(self.count_reg_pts) == 2:
                    if (prev_position is not None and track_id not in self.
                        count_ids):
                        distance = Point(track_line[-1]).distance(self.
                            counting_region)
                        if (distance < self.line_dist_thresh and track_id
                             not in self.count_ids):
                            self.count_ids.append(track_id)
                            if (box[0] - prev_position[0]) * (self.
                                counting_region.centroid.x - prev_position[0]
                                ) > 0:
                                self.in_counts += 1
                                self.class_wise_count[self.names[cls]]['IN'
                                    ] += 1
                            else:
                                self.out_counts += 1
                                self.class_wise_count[self.names[cls]]['OUT'
                                    ] += 1
    else:
        for box, cls in zip(self.boxes, self.clss):
            if self.shape == 'circle':
                center = int((box[0] + box[2]) // 2), int((box[1] + box[3]) //
                    2)
                radius = min(int(box[2]) - int(box[0]), int(box[3]) - int(
                    box[1])) // 2
                y, x = np.ogrid[0:self.heatmap.shape[0], 0:self.heatmap.
                    shape[1]]
                mask = (x - center[0]) ** 2 + (y - center[1]
                    ) ** 2 <= radius ** 2
                self.heatmap[int(box[1]):int(box[3]), int(box[0]):int(box[2])
                    ] += 2 * mask[int(box[1]):int(box[3]), int(box[0]):int(
                    box[2])]
            else:
                self.heatmap[int(box[1]):int(box[3]), int(box[0]):int(box[2])
                    ] += 2
    if self.count_reg_pts is not None:
        labels_dict = {}
        for key, value in self.class_wise_count.items():
            if value['IN'] != 0 or value['OUT'] != 0:
                if not self.view_in_counts and not self.view_out_counts:
                    continue
                elif not self.view_in_counts:
                    labels_dict[str.capitalize(key)] = f"OUT {value['OUT']}"
                elif not self.view_out_counts:
                    labels_dict[str.capitalize(key)] = f"IN {value['IN']}"
                else:
                    labels_dict[str.capitalize(key)
                        ] = f"IN {value['IN']} OUT {value['OUT']}"
        if labels_dict is not None:
            self.annotator.display_analytics(self.im0, labels_dict, self.
                count_txt_color, self.count_bg_color, 10)
    heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.
        NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8),
        self.colormap)
    self.im0 = cv2.addWeighted(self.im0, 1 - self.heatmap_alpha,
        heatmap_colored, self.heatmap_alpha, 0)
    if self.env_check and self.view_img:
        self.display_frames()
    return self.im0
