def start_counting(self, im0, results):
    """
        Function used to count the gym steps.

        Args:
            im0 (ndarray): Current frame from the video stream.
            results (list): Pose estimation data.
        """
    self.im0 = im0
    if not len(results[0]):
        return self.im0
    if len(results[0]) > len(self.count):
        new_human = len(results[0]) - len(self.count)
        self.count += [0] * new_human
        self.angle += [0] * new_human
        self.stage += ['-'] * new_human
    self.keypoints = results[0].keypoints.data
    self.annotator = Annotator(im0, line_width=self.tf)
    for ind, k in enumerate(reversed(self.keypoints)):
        if self.pose_type in {'pushup', 'pullup', 'abworkout', 'squat'}:
            self.angle[ind] = self.annotator.estimate_pose_angle(k[int(self
                .kpts_to_check[0])].cpu(), k[int(self.kpts_to_check[1])].
                cpu(), k[int(self.kpts_to_check[2])].cpu())
            self.im0 = self.annotator.draw_specific_points(k, self.
                kpts_to_check, shape=(640, 640), radius=10)
            if self.pose_type in {'abworkout', 'pullup'}:
                if self.angle[ind] > self.poseup_angle:
                    self.stage[ind] = 'down'
                if self.angle[ind] < self.posedown_angle and self.stage[ind
                    ] == 'down':
                    self.stage[ind] = 'up'
                    self.count[ind] += 1
            elif self.pose_type in {'pushup', 'squat'}:
                if self.angle[ind] > self.poseup_angle:
                    self.stage[ind] = 'up'
                if self.angle[ind] < self.posedown_angle and self.stage[ind
                    ] == 'up':
                    self.stage[ind] = 'down'
                    self.count[ind] += 1
            self.annotator.plot_angle_and_count_and_stage(angle_text=self.
                angle[ind], count_text=self.count[ind], stage_text=self.
                stage[ind], center_kpt=k[int(self.kpts_to_check[1])])
        self.annotator.kpts(k, shape=(640, 640), radius=1, kpt_line=True)
    if self.env_check and self.view_img:
        cv2.imshow('Ultralytics YOLOv8 AI GYM', self.im0)
        if cv2.waitKey(1) & 255 == ord('q'):
            return
    return self.im0
