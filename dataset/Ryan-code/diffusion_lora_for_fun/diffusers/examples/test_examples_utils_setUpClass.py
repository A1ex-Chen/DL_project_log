@classmethod
def setUpClass(cls):
    super().setUpClass()
    cls._tmpdir = tempfile.mkdtemp()
    cls.configPath = os.path.join(cls._tmpdir, 'default_config.yml')
    write_basic_config(save_location=cls.configPath)
    cls._launch_args = ['accelerate', 'launch', '--config_file', cls.configPath
        ]
