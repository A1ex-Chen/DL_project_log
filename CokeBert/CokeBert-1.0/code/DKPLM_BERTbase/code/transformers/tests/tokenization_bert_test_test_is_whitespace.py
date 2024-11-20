def test_is_whitespace(self):
    self.assertTrue(_is_whitespace(u' '))
    self.assertTrue(_is_whitespace(u'\t'))
    self.assertTrue(_is_whitespace(u'\r'))
    self.assertTrue(_is_whitespace(u'\n'))
    self.assertTrue(_is_whitespace(u'\xa0'))
    self.assertFalse(_is_whitespace(u'A'))
    self.assertFalse(_is_whitespace(u'-'))
