def test_is_control(self):
    self.assertTrue(_is_control(u'\x05'))
    self.assertFalse(_is_control(u'A'))
    self.assertFalse(_is_control(u' '))
    self.assertFalse(_is_control(u'\t'))
    self.assertFalse(_is_control(u'\r'))
