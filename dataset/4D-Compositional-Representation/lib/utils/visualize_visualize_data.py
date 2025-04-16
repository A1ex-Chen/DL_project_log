def visualize_data(data, data_type, out_file):
    """ Visualizes the data with regard to its type.

    Args:
        data (tensor): batch of data
        data_type (string): data type (img, voxels or pointcloud)
        out_file (string): output file
    """
    if data_type == 'img':
        if data.dim() == 3:
            data = data.unsqueeze(0)
        save_image(data, out_file, nrow=4)
    elif data_type == 'voxels':
        visualize_voxels(data, out_file=out_file)
    elif data_type == 'pointcloud':
        visualize_pointcloud(data, out_file=out_file)
    elif data_type is None or data_type == 'idx':
        pass
    else:
        raise ValueError('Invalid data_type "%s"' % data_type)
