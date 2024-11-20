def download_file_from_google_drive(id, destination):
    URL = 'https://docs.google.com/uc?export=download&confirm=1'
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    print(URL, id)
    save_response_content(response, destination)
