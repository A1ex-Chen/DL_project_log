def extract_and_process_tracks(self, tracks):
    """Extracts and processes tracks for object counting in a video stream."""
    self.annotator = Annotator(self.im0, self.tf, self.names)
    self.annotator.draw_region(reg_pts=self.reg_pts, color=self.
        region_color, thickness=self.region_thickness)
    if tracks[0].boxes.id is not None:
        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        track_ids = tracks[0].boxes.id.int().cpu().tolist()
        for box, track_id, cls in zip(boxes, track_ids, clss):
            self.annotator.box_label(box, label=
                f'{self.names[cls]}#{track_id}', color=colors(int(track_id),
                True))
            if self.names[cls] not in self.class_wise_count:
                self.class_wise_count[self.names[cls]] = {'IN': 0, 'OUT': 0}
            track_line = self.track_history[track_id]
            track_line.append((float((box[0] + box[2]) / 2), float((box[1] +
                box[3]) / 2)))
            if len(track_line) > 30:
                track_line.pop(0)
            if self.draw_tracks:
                self.annotator.draw_centroid_and_tracks(track_line, color=
                    self.track_color or colors(int(track_id), True),
                    track_thickness=self.track_thickness)
            prev_position = self.track_history[track_id][-2] if len(self.
                track_history[track_id]) > 1 else None
            if len(self.reg_pts) >= 3:
                is_inside = self.counting_region.contains(Point(track_line[-1])
                    )
                if (prev_position is not None and is_inside and track_id not in
                    self.count_ids):
                    self.count_ids.append(track_id)
                    if (box[0] - prev_position[0]) * (self.counting_region.
                        centroid.x - prev_position[0]) > 0:
                        self.in_counts += 1
                        self.class_wise_count[self.names[cls]]['IN'] += 1
                    else:
                        self.out_counts += 1
                        self.class_wise_count[self.names[cls]]['OUT'] += 1
            elif len(self.reg_pts) == 2:
                if (prev_position is not None and track_id not in self.
                    count_ids):
                    distance = Point(track_line[-1]).distance(self.
                        counting_region)
                    if (distance < self.line_dist_thresh and track_id not in
                        self.count_ids):
                        self.count_ids.append(track_id)
                        if (box[0] - prev_position[0]) * (self.
                            counting_region.centroid.x - prev_position[0]) > 0:
                            self.in_counts += 1
                            self.class_wise_count[self.names[cls]]['IN'] += 1
                        else:
                            self.out_counts += 1
                            self.class_wise_count[self.names[cls]]['OUT'] += 1
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
    if labels_dict:
        self.annotator.display_analytics(self.im0, labels_dict, self.
            count_txt_color, self.count_bg_color, 10)
