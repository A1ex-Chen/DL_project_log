def run(self):
    token = HfFolder.get_token()
    if token is None:
        print('Not logged in')
        exit(1)
    local_path = os.path.abspath(self.args.path)
    if os.path.isdir(local_path):
        if self.args.filename is not None:
            raise ValueError(
                'Cannot specify a filename override when uploading a folder.')
        rel_path = os.path.basename(local_path)
        files = self.walk_dir(rel_path)
    elif os.path.isfile(local_path):
        filename = (self.args.filename if self.args.filename is not None else
            os.path.basename(local_path))
        files = [(local_path, filename)]
    else:
        raise ValueError('Not a valid file or directory: {}'.format(local_path)
            )
    if sys.platform == 'win32':
        files = [(filepath, filename.replace(os.sep, '/')) for filepath,
            filename in files]
    if len(files) > UPLOAD_MAX_FILES:
        print(
            'About to upload {} files to S3. This is probably wrong. Please filter files before uploading.'
            .format(ANSI.bold(len(files))))
        exit(1)
    user, _ = self._api.whoami(token)
    namespace = (self.args.organization if self.args.organization is not
        None else user)
    for filepath, filename in files:
        print(
            'About to upload file {} to S3 under filename {} and namespace {}'
            .format(ANSI.bold(filepath), ANSI.bold(filename), ANSI.bold(
            namespace)))
    if not self.args.yes:
        choice = input('Proceed? [Y/n] ').lower()
        if not (choice == '' or choice == 'y' or choice == 'yes'):
            print('Abort')
            exit()
    print(ANSI.bold('Uploading... This might take a while if files are large'))
    for filepath, filename in files:
        try:
            access_url = self._api.presign_and_upload(token=token, filename
                =filename, filepath=filepath, organization=self.args.
                organization)
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        print('Your file now lives at:')
        print(access_url)
