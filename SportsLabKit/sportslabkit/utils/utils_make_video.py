def make_video(frames: Iterable[NDArray[np.uint8]], outpath: PathLike,
    vcodec: str='libx264', pix_fmt: str='yuv420p', preset: str='medium',
    crf: (int | None)=None, ss: (int | None)=None, t: (int | None)=None, c:
    (str | None)=None, height: (int | None)=-1, width: (int | None)=-1,
    input_framerate: (int | None)=None, logging: bool=False, custom_ffmpeg:
    (str | None)=None) ->None:
    """Make video from a list of opencv format frames.

    Args:
        frames (Iterable[NDArray[np.uint8]]): List of opencv format frames
        outpath (str): Path to output video file
        vcodec (str): Video codec.
        preset (str): Video encoding preset. A preset is a collection of options
            that will provide a certain encoding speed to compression ratio. A
            slower preset will provide better compression (compression is quality
            per filesize). Use the slowest preset that you have patience for.
            The available presets in descending order of speed are:

            - ultrafast
            - superfast
            - veryfast
            - faster
            - fast
            - medium (default preset)
            - slow
            - slower
            - veryslow

            Defaults to `medium`.

        crf (int): Constant Rate Factor. Use the crf (Constant Rate Factor)
            parameter to control the output quality. The lower crf, the higher
            the quality (range: 0-51). Visually lossless compression corresponds
            to -crf 18. Use the preset parameter to control the speed of the
            compression process. Defaults to `23`.
        ss (int): Start-time of the clip in seconds. Defaults to `0`.
        t (Optional[int]): Duration of the clip in seconds. Defaults to None.
        c (bool): copies the first video, audio, and subtitle bitstream from the input to the output file without re-encoding them. Defaults to `False`.
        height (int): Video height. Defaults to `None`.
        width (int): Video width. Defaults to `None`.
        input_framerate (int): Input framerate. Defaults to `25`.
        logging (bool): Logging. Defaults to `False`.
    Todo:
        * add FPS option
        * functionality to use PIL image
        * reconsider compression (current compression is not good)
    """
    scale_filter = f'scale={width}:{height}'
    output_params = {k: v for k, v in {'-vcodec': vcodec, '-pix_fmt':
        pix_fmt, '-crf': crf, '-preset': preset, '-vf': scale_filter, '-c':
        c, '-ss': ss, '-t': t, '-input_framerate': input_framerate}.items() if
        v is not None}
    logger.debug(f'output_params: {output_params}')
    if not Path(outpath).parent.exists():
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
    writer = WriteGear(output=outpath, compression_mode=True, logging=
        logging, custom_ffmpeg=custom_ffmpeg, **output_params)
    for frame in tqdm(frames, desc='Writing video', level='INFO'):
        writer.write(frame, rgb_mode=True)
    writer.close()
