def test_set_log_level(self):
    set_log_level('DEBUG')
    self.assertEqual(level_filter.level, 'DEBUG')
    self.assertEqual(os.environ['LOG_LEVEL'], 'DEBUG')
