def draw_sem_seg(self, frame, sem_seg, area_threshold=None):
    """
        Args:
            sem_seg (ndarray or Tensor): semantic segmentation of shape (H, W),
                each value is the integer label.
            area_threshold (Optional[int]): only draw segmentations larger than the threshold
        """
    frame_visualizer = Visualizer(frame, self.metadata)
    frame_visualizer.draw_sem_seg(sem_seg, area_threshold=None)
    return frame_visualizer.output
