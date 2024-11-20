def test_is_punctuation(self):
    self.assertTrue(_is_punctuation(u'-'))
    self.assertTrue(_is_punctuation(u'$'))
    self.assertTrue(_is_punctuation(u'`'))
    self.assertTrue(_is_punctuation(u'.'))
    self.assertFalse(_is_punctuation(u'A'))
    self.assertFalse(_is_punctuation(u' '))
