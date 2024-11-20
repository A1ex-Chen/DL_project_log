def export_to_gif(image: List[PIL.Image.Image], output_gif_path: str=None,
    fps: int=10) ->str:
    if output_gif_path is None:
        output_gif_path = tempfile.NamedTemporaryFile(suffix='.gif').name
    image[0].save(output_gif_path, save_all=True, append_images=image[1:],
        optimize=False, duration=1000 // fps, loop=0)
    return output_gif_path
