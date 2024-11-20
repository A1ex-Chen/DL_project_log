@ThreadingLocked()
def check_font(font='Arial.ttf'):
    """
    Find font locally or download to user's configuration directory if it does not already exist.

    Args:
        font (str): Path or name of font.

    Returns:
        file (Path): Resolved font file path.
    """
    from matplotlib import font_manager
    name = Path(font).name
    file = USER_CONFIG_DIR / name
    if file.exists():
        return file
    matches = [s for s in font_manager.findSystemFonts() if font in s]
    if any(matches):
        return matches[0]
    url = (
        f'https://github.com/ultralytics/assets/releases/download/v0.0.0/{name}'
        )
    if downloads.is_url(url, check=True):
        downloads.safe_download(url=url, file=file)
        return file
