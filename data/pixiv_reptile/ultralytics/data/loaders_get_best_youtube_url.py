def get_best_youtube_url(url, method='pytube'):
    """
    Retrieves the URL of the best quality MP4 video stream from a given YouTube video.

    This function uses the specified method to extract the video info from YouTube. It supports the following methods:
    - "pytube": Uses the pytube library to fetch the video streams.
    - "pafy": Uses the pafy library to fetch the video streams.
    - "yt-dlp": Uses the yt-dlp library to fetch the video streams.

    The function then finds the highest quality MP4 format that has a video codec but no audio codec, and returns the
    URL of this video stream.

    Args:
        url (str): The URL of the YouTube video.
        method (str): The method to use for extracting video info. Default is "pytube". Other options are "pafy" and
            "yt-dlp".

    Returns:
        (str): The URL of the best quality MP4 video stream, or None if no suitable stream is found.
    """
    if method == 'pytube':
        check_requirements('pytubefix')
        from pytubefix import YouTube
        streams = YouTube(url).streams.filter(file_extension='mp4',
            only_video=True)
        streams = sorted(streams, key=lambda s: s.resolution, reverse=True)
        for stream in streams:
            if stream.resolution and int(stream.resolution[:-1]) >= 1080:
                return stream.url
    elif method == 'pafy':
        check_requirements(('pafy', 'youtube_dl==2020.12.2'))
        import pafy
        return pafy.new(url).getbestvideo(preftype='mp4').url
    elif method == 'yt-dlp':
        check_requirements('yt-dlp')
        import yt_dlp
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)
        for f in reversed(info_dict.get('formats', [])):
            good_size = (f.get('width') or 0) >= 1920 or (f.get('height') or 0
                ) >= 1080
            if good_size and f['vcodec'] != 'none' and f['acodec'
                ] == 'none' and f['ext'] == 'mp4':
                return f.get('url')
