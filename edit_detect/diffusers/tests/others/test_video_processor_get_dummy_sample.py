def get_dummy_sample(self, input_type):
    batch_size = 1
    num_frames = 5
    num_channels = 3
    height = 8
    width = 8

    def generate_image():
        return PIL.Image.fromarray(np.random.randint(0, 256, size=(height,
            width, num_channels)).astype('uint8'))

    def generate_4d_array():
        return np.random.rand(num_frames, height, width, num_channels)

    def generate_5d_array():
        return np.random.rand(batch_size, num_frames, height, width,
            num_channels)

    def generate_4d_tensor():
        return torch.rand(num_frames, num_channels, height, width)

    def generate_5d_tensor():
        return torch.rand(batch_size, num_frames, num_channels, height, width)
    if input_type == 'list_images':
        sample = [generate_image() for _ in range(num_frames)]
    elif input_type == 'list_list_images':
        sample = [[generate_image() for _ in range(num_frames)] for _ in
            range(num_frames)]
    elif input_type == 'list_4d_np':
        sample = [generate_4d_array() for _ in range(num_frames)]
    elif input_type == 'list_list_4d_np':
        sample = [[generate_4d_array() for _ in range(num_frames)] for _ in
            range(num_frames)]
    elif input_type == 'list_5d_np':
        sample = [generate_5d_array() for _ in range(num_frames)]
    elif input_type == '5d_np':
        sample = generate_5d_array()
    elif input_type == 'list_4d_pt':
        sample = [generate_4d_tensor() for _ in range(num_frames)]
    elif input_type == 'list_list_4d_pt':
        sample = [[generate_4d_tensor() for _ in range(num_frames)] for _ in
            range(num_frames)]
    elif input_type == 'list_5d_pt':
        sample = [generate_5d_tensor() for _ in range(num_frames)]
    elif input_type == '5d_pt':
        sample = generate_5d_tensor()
    return sample
