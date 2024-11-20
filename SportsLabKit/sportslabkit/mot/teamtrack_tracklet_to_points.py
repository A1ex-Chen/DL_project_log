def tracklet_to_points(self, tracklet, H):
    boxes = np.array(tracklet.get_observations('box'))
    bcxs, bcys = boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3]
    x = cv2.perspectiveTransform(np.stack([bcxs, bcys], axis=1).reshape(1, 
        -1, 2).astype('float32'), H)[0]
    x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x)
    return x
