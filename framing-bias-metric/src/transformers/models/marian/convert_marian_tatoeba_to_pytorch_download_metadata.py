def download_metadata(self):
    Path(LANG_CODE_PATH).parent.mkdir(exist_ok=True)
    import wget
    if not os.path.exists(ISO_PATH):
        wget.download(ISO_URL, ISO_PATH)
    if not os.path.exists(LANG_CODE_PATH):
        wget.download(LANG_CODE_URL, LANG_CODE_PATH)
