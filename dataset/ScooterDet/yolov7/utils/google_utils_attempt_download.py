def attempt_download(file, repo='WongKinYiu/yolov7'):
    file = Path(str(file).strip().replace("'", '').lower())
    if not file.exists():
        try:
            response = requests.get(
                f'https://api.github.com/repos/{repo}/releases/latest').json()
            assets = [x['name'] for x in response['assets']]
            tag = response['tag_name']
        except:
            assets = ['yolov7.pt', 'yolov7-tiny.pt', 'yolov7x.pt',
                'yolov7-d6.pt', 'yolov7-e6.pt', 'yolov7-e6e.pt', 'yolov7-w6.pt'
                ]
            tag = subprocess.check_output('git tag', shell=True).decode(
                ).split()[-1]
        name = file.name
        if name in assets:
            msg = (
                f'{file} missing, try downloading from https://github.com/{repo}/releases/'
                )
            redundant = False
            try:
                url = (
                    f'https://github.com/{repo}/releases/download/{tag}/{name}'
                    )
                print(f'Downloading {url} to {file}...')
                torch.hub.download_url_to_file(url, file)
                assert file.exists() and file.stat().st_size > 1000000.0
            except Exception as e:
                print(f'Download error: {e}')
                assert redundant, 'No secondary mirror'
                url = f'https://storage.googleapis.com/{repo}/ckpt/{name}'
                print(f'Downloading {url} to {file}...')
                os.system(f'curl -L {url} -o {file}')
            finally:
                if not file.exists() or file.stat().st_size < 1000000.0:
                    file.unlink(missing_ok=True)
                    print(f'ERROR: Download failure: {msg}')
                print('')
                return
