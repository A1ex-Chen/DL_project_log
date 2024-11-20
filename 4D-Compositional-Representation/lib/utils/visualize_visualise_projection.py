def visualise_projection(self, points, world_mat, camera_mat, img,
    output_file='out.png'):
    """ Visualizes the transformation and projection to image plane.

        The first points of the batch are transformed and projected to the
        respective image. After performing the relevant transformations, the
        visualization is saved in the provided output_file path.

    Arguments:
        points (tensor): batch of point cloud points
        world_mat (tensor): batch of matrices to rotate pc to camera-based
                coordinates
        camera_mat (tensor): batch of camera matrices to project to 2D image
                plane
        img (tensor): tensor of batch GT image files
        output_file (string): where the output should be saved
    """
    points_transformed = common.transform_points(points, world_mat)
    points_img = common.project_to_camera(points_transformed, camera_mat)
    pimg2 = points_img[0].detach().cpu().numpy()
    image = img[0].cpu().numpy()
    plt.imshow(image.transpose(1, 2, 0))
    plt.plot((pimg2[:, 0] + 1) * image.shape[1] / 2, (pimg2[:, 1] + 1) *
        image.shape[2] / 2, 'x')
    plt.savefig(output_file)
