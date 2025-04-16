def test_auto_string_parser(self):
    for value in [True, False, 500, 50.0, 'Hellow World']:
        s = f'{value}'
        parsed_value = auto_string_parser(s)
        self.assertEqual(parsed_value, value)
