@staticmethod
def register_subcommand(parser: ArgumentParser):
    login_parser = parser.add_parser('login', help=
        'Log in using the same credentials as on huggingface.co')
    login_parser.set_defaults(func=lambda args: LoginCommand(args))
    whoami_parser = parser.add_parser('whoami', help=
        'Find out which huggingface.co account you are logged in as.')
    whoami_parser.set_defaults(func=lambda args: WhoamiCommand(args))
    logout_parser = parser.add_parser('logout', help='Log out')
    logout_parser.set_defaults(func=lambda args: LogoutCommand(args))
    s3_parser = parser.add_parser('s3_datasets', help=
        '{ls, rm} Commands to interact with the files you upload on S3.')
    s3_subparsers = s3_parser.add_subparsers(help='s3 related commands')
    ls_parser = s3_subparsers.add_parser('ls')
    ls_parser.add_argument('--organization', type=str, help=
        'Optional: organization namespace.')
    ls_parser.set_defaults(func=lambda args: ListObjsCommand(args))
    rm_parser = s3_subparsers.add_parser('rm')
    rm_parser.add_argument('filename', type=str, help=
        'individual object filename to delete from huggingface.co.')
    rm_parser.add_argument('--organization', type=str, help=
        'Optional: organization namespace.')
    rm_parser.set_defaults(func=lambda args: DeleteObjCommand(args))
    upload_parser = s3_subparsers.add_parser('upload', help=
        'Upload a file to S3.')
    upload_parser.add_argument('path', type=str, help=
        'Local path of the folder or individual file to upload.')
    upload_parser.add_argument('--organization', type=str, help=
        'Optional: organization namespace.')
    upload_parser.add_argument('--filename', type=str, default=None, help=
        'Optional: override individual object filename on S3.')
    upload_parser.add_argument('-y', '--yes', action='store_true', help=
        'Optional: answer Yes to the prompt')
    upload_parser.set_defaults(func=lambda args: UploadCommand(args))
    upload_parser = parser.add_parser('upload', help=
        'Deprecated: used to be the way to upload a model to S3. We now use a git-based system for storing models and other artifacts. Use the `repo create` command instead.'
        )
    upload_parser.set_defaults(func=lambda args: DeprecatedUploadCommand(args))
    repo_parser = parser.add_parser('repo', help=
        '{create, ls-files} Commands to interact with your huggingface.co repos.'
        )
    repo_subparsers = repo_parser.add_subparsers(help=
        'huggingface.co repos related commands')
    ls_parser = repo_subparsers.add_parser('ls-files', help=
        'List all your files on huggingface.co')
    ls_parser.add_argument('--organization', type=str, help=
        'Optional: organization namespace.')
    ls_parser.set_defaults(func=lambda args: ListReposObjsCommand(args))
    repo_create_parser = repo_subparsers.add_parser('create', help=
        'Create a new repo on huggingface.co')
    repo_create_parser.add_argument('name', type=str, help=
        "Name for your model's repo. Will be namespaced under your username to build the model id."
        )
    repo_create_parser.add_argument('--organization', type=str, help=
        'Optional: organization namespace.')
    repo_create_parser.add_argument('-y', '--yes', action='store_true',
        help='Optional: answer Yes to the prompt')
    repo_create_parser.set_defaults(func=lambda args: RepoCreateCommand(args))
