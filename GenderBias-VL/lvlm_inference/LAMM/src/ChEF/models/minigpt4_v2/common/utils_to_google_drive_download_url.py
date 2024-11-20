def to_google_drive_download_url(view_url: str) ->str:
    """
    Utility function to transform a view URL of google drive
    to a download URL for google drive
    Example input:
        https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp/view
    Example output:
        https://drive.google.com/uc?export=download&id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp
    """
    splits = view_url.split('/')
    assert splits[-1] == 'view'
    file_id = splits[-2]
    return f'https://drive.google.com/uc?export=download&id={file_id}'
