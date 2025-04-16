def to_np(self, video):
    if isinstance(video[0], PIL.Image.Image):
        video = np.stack([np.array(i) for i in video], axis=0)
    elif isinstance(video, list) and isinstance(video[0][0], PIL.Image.Image):
        frames = []
        for vid in video:
            all_current_frames = np.stack([np.array(i) for i in vid], axis=0)
            frames.append(all_current_frames)
        video = np.stack([np.array(frame) for frame in frames], axis=0)
    elif isinstance(video, list) and isinstance(video[0], (torch.Tensor, np
        .ndarray)):
        if isinstance(video[0], np.ndarray):
            video = np.stack(video, axis=0) if video[0
                ].ndim == 4 else np.concatenate(video, axis=0)
        elif video[0].ndim == 4:
            video = np.stack([i.cpu().numpy().transpose(0, 2, 3, 1) for i in
                video], axis=0)
        elif video[0].ndim == 5:
            video = np.concatenate([i.cpu().numpy().transpose(0, 1, 3, 4, 2
                ) for i in video], axis=0)
    elif isinstance(video, list) and isinstance(video[0], list) and isinstance(
        video[0][0], (torch.Tensor, np.ndarray)):
        all_frames = []
        for list_of_videos in video:
            temp_frames = []
            for vid in list_of_videos:
                if vid.ndim == 4:
                    current_vid_frames = np.stack([(i if isinstance(i, np.
                        ndarray) else i.cpu().numpy().transpose(1, 2, 0)) for
                        i in vid], axis=0)
                elif vid.ndim == 5:
                    current_vid_frames = np.concatenate([(i if isinstance(i,
                        np.ndarray) else i.cpu().numpy().transpose(0, 2, 3,
                        1)) for i in vid], axis=0)
                temp_frames.append(current_vid_frames)
            temp_frames = np.stack(temp_frames, axis=0)
            all_frames.append(temp_frames)
        video = np.concatenate(all_frames, axis=0)
    elif isinstance(video, (torch.Tensor, np.ndarray)) and video.ndim == 5:
        video = video if isinstance(video, np.ndarray) else video.cpu().numpy(
            ).transpose(0, 1, 3, 4, 2)
    return video
