def get_google_drive_file_info(link):
    """
    Retrieves the direct download link and filename for a shareable Google Drive file link.

    Args:
        link (str): The shareable link of the Google Drive file.

    Returns:
        (str): Direct download URL for the Google Drive file.
        (str): Original filename of the Google Drive file. If filename extraction fails, returns None.

    Example:
        ```python
        from ultralytics.utils.downloads import get_google_drive_file_info

        link = "https://drive.google.com/file/d/1cqT-cJgANNrhIHCrEufUYhQ4RqiWG_lJ/view?usp=drive_link"
        url, filename = get_google_drive_file_info(link)
        ```
    """
    file_id = link.split('/d/')[1].split('/view')[0]
    drive_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    filename = None
    with requests.Session() as session:
        response = session.get(drive_url, stream=True)
        if 'quota exceeded' in str(response.content.lower()):
            raise ConnectionError(emojis(
                f'❌  Google Drive file download quota exceeded. Please try again later or download this file manually at {link}.'
                ))
        for k, v in response.cookies.items():
            if k.startswith('download_warning'):
                drive_url += f'&confirm={v}'
        cd = response.headers.get('content-disposition')
        if cd:
            filename = re.findall('filename="(.+)"', cd)[0]
    return drive_url, filename
