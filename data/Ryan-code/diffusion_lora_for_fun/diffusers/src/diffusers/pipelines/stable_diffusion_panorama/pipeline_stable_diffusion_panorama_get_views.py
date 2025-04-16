def get_views(self, panorama_height: int, panorama_width: int, window_size:
    int=64, stride: int=8, circular_padding: bool=False) ->List[Tuple[int,
    int, int, int]]:
    """
        Generates a list of views based on the given parameters. Here, we define the mappings F_i (see Eq. 7 in the
        MultiDiffusion paper https://arxiv.org/abs/2302.08113). If panorama's height/width < window_size, num_blocks of
        height/width should return 1.

        Args:
            panorama_height (int): The height of the panorama.
            panorama_width (int): The width of the panorama.
            window_size (int, optional): The size of the window. Defaults to 64.
            stride (int, optional): The stride value. Defaults to 8.
            circular_padding (bool, optional): Whether to apply circular padding. Defaults to False.

        Returns:
            List[Tuple[int, int, int, int]]: A list of tuples representing the views. Each tuple contains four integers
            representing the start and end coordinates of the window in the panorama.

        """
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size
        ) // stride + 1 if panorama_height > window_size else 1
    if circular_padding:
        num_blocks_width = (panorama_width // stride if panorama_width >
            window_size else 1)
    else:
        num_blocks_width = (panorama_width - window_size
            ) // stride + 1 if panorama_width > window_size else 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int(i // num_blocks_width * stride)
        h_end = h_start + window_size
        w_start = int(i % num_blocks_width * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views
