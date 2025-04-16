def create_rotation_matrix(self, offset=0):
    center = self.center[0] + offset, self.center[1] + offset
    rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
    if self.expand:
        rot_im_center = cv2.transform(self.image_center[None, None, :] +
            offset, rm)[0, 0, :]
        new_center = np.array([self.bound_w / 2, self.bound_h / 2]
            ) + offset - rot_im_center
        rm[:, 2] += new_center
    return rm
