def extract_and_process_tracks(self, tracks):
    """Extracts and processes tracks for queue management in a video stream."""
    self.annotator = Annotator(self.im0, self.tf, self.names)
    if tracks[0].boxes.id is not None:
        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        track_ids = tracks[0].boxes.id.int().cpu().tolist()
        for box, track_id, cls in zip(boxes, track_ids, clss):
            self.annotator.box_label(box, label=
                f'{self.names[cls]}#{track_id}', color=colors(int(track_id),
                True))
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
                if prev_position is not None and is_inside:
                    self.counts += 1
    label = f'Queue Counts : {str(self.counts)}'
    if label is not None:
        self.annotator.queue_counts_display(label, points=self.reg_pts,
            region_color=self.region_color, txt_color=self.count_txt_color)
    self.counts = 0
    self.display_frames()
