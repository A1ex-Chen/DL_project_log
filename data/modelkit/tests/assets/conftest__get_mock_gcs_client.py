def _get_mock_gcs_client():
    my_http = requests.Session()
    my_http.verify = False
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return storage.Client(credentials=AnonymousCredentials(), project=
        'test', _http=my_http, client_options=ClientOptions(api_endpoint=
        'https://127.0.0.1:4443'))
