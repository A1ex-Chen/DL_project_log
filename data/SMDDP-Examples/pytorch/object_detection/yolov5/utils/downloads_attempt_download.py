def attempt_download(file, repo='ultralytics/yolov5', release='v6.1'):
    from utils.general import LOGGER

    def github_assets(repository, version='latest'):
        if version != 'latest':
            version = f'tags/{version}'
        response = requests.get(
            f'https://api.github.com/repos/{repository}/releases/{version}'
            ).json()
        return response['tag_name'], [x['name'] for x in response['assets']]
    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():
        name = Path(urllib.parse.unquote(str(file))).name
        if str(file).startswith(('http:/', 'https:/')):
            url = str(file).replace(':/', '://')
            file = name.split('?')[0]
            if Path(file).is_file():
                LOGGER.info(f'Found {url} locally at {file}')
            else:
                safe_download(file=file, url=url, min_bytes=100000.0)
            return file
        assets = ['yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt',
            'yolov5x.pt', 'yolov5n6.pt', 'yolov5s6.pt', 'yolov5m6.pt',
            'yolov5l6.pt', 'yolov5x6.pt']
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)
            except Exception:
                try:
                    tag = subprocess.check_output('git tag', shell=True,
                        stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release
        file.parent.mkdir(parents=True, exist_ok=True)
        if name in assets:
            url3 = (
                'https://drive.google.com/drive/folders/1EFQTEUeXWSFww0luse2jB9M1QNZQGwNl'
                )
            safe_download(file, url=
                f'https://github.com/{repo}/releases/download/{tag}/{name}',
                url2=f'https://storage.googleapis.com/{repo}/{tag}/{name}',
                min_bytes=100000.0, error_msg=
                f'{file} missing, try downloading from https://github.com/{repo}/releases/{tag} or {url3}'
                )
    return str(file)
