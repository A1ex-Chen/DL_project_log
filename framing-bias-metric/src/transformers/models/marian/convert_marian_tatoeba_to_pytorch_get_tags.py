def get_tags(self, code, ref_name):
    if len(code) == 2:
        assert 'languages' not in ref_name, f'{code}: {ref_name}'
        return [code], False
    elif 'languages' in ref_name or len(self.constituents.get(code, [])) > 1:
        group = self.expand_group_to_two_letter_codes(code)
        group.append(code)
        return group, True
    else:
        print(f'Three letter monolingual code: {code}')
        return [code], False
