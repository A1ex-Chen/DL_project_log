def plot_angle_and_count_and_stage(self, angle_text, count_text, stage_text,
    center_kpt, color=(104, 31, 17), txt_color=(255, 255, 255)):
    """
        Plot the pose angle, count value and step stage.

        Args:
            angle_text (str): angle value for workout monitoring
            count_text (str): counts value for workout monitoring
            stage_text (str): stage decision for workout monitoring
            center_kpt (list): centroid pose index for workout monitoring
            color (tuple): text background color for workout monitoring
            txt_color (tuple): text foreground color for workout monitoring
        """
    angle_text, count_text, stage_text = (f' {angle_text:.2f}',
        f'Steps : {count_text}', f' {stage_text}')
    (angle_text_width, angle_text_height), _ = cv2.getTextSize(angle_text, 
        0, self.sf, self.tf)
    angle_text_position = int(center_kpt[0]), int(center_kpt[1])
    angle_background_position = angle_text_position[0], angle_text_position[1
        ] - angle_text_height - 5
    angle_background_size = (angle_text_width + 2 * 5, angle_text_height + 
        2 * 5 + self.tf * 2)
    cv2.rectangle(self.im, angle_background_position, (
        angle_background_position[0] + angle_background_size[0], 
        angle_background_position[1] + angle_background_size[1]), color, -1)
    cv2.putText(self.im, angle_text, angle_text_position, 0, self.sf,
        txt_color, self.tf)
    (count_text_width, count_text_height), _ = cv2.getTextSize(count_text, 
        0, self.sf, self.tf)
    count_text_position = angle_text_position[0], angle_text_position[1
        ] + angle_text_height + 20
    count_background_position = angle_background_position[0
        ], angle_background_position[1] + angle_background_size[1] + 5
    count_background_size = (count_text_width + 10, count_text_height + 10 +
        self.tf)
    cv2.rectangle(self.im, count_background_position, (
        count_background_position[0] + count_background_size[0], 
        count_background_position[1] + count_background_size[1]), color, -1)
    cv2.putText(self.im, count_text, count_text_position, 0, self.sf,
        txt_color, self.tf)
    (stage_text_width, stage_text_height), _ = cv2.getTextSize(stage_text, 
        0, self.sf, self.tf)
    stage_text_position = int(center_kpt[0]), int(center_kpt[1]
        ) + angle_text_height + count_text_height + 40
    stage_background_position = stage_text_position[0], stage_text_position[1
        ] - stage_text_height - 5
    stage_background_size = stage_text_width + 10, stage_text_height + 10
    cv2.rectangle(self.im, stage_background_position, (
        stage_background_position[0] + stage_background_size[0], 
        stage_background_position[1] + stage_background_size[1]), color, -1)
    cv2.putText(self.im, stage_text, stage_text_position, 0, self.sf,
        txt_color, self.tf)
