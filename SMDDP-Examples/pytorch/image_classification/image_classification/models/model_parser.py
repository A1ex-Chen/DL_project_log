def parser(self):
    if self.model.params is None:
        return None
    parser = self.model.params.parser(self.name)
    parser.add_argument('--pretrained-from-file', default=None, type=str,
        metavar='PATH', help='load weights from local file')
    if self.model.checkpoint_url is not None:
        parser.add_argument('--pretrained', default=False, action=
            'store_true', help='load pretrained weights from NGC')
    return parser
