@staticmethod
def from_bbdf(bbdf):
    precomputed_detections = []
    cols = ['bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class']
    for frame_idx, frame_df in bbdf.iter_frames():
        long_df = frame_df.to_long_df()
        long_df['class'] = 0
        d = long_df[cols].rename(columns={'bb_left': 'bbox_left', 'bb_top':
            'bbox_top', 'bb_width': 'bbox_width', 'bb_height': 'bbox_height'}
            ).to_dict(orient='records')
        precomputed_detections.append(d)
    return DummyDetectionModel(precomputed_detections)
