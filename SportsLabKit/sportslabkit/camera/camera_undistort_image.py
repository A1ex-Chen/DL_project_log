def undistort_image(self, image: NDArray) ->NDArray:
    undistorted_image = cv.remap(image, self.mapx, self.mapy, interpolation
        =cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    return undistorted_image
