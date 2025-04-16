def get_string_spec(versions):
    examples = [('blabli/blebla', {'name': 'blabli/blebla'}), (
        'bl2_32_a.bli/bl3.debla', {'name': 'bl2_32_a.bli/bl3.debla'}), (
        'blabli/BLEBLA', {'name': 'blabli/BLEBLA'}), (
        'bl2_32_a.BLI/bl3.DEBLA', {'name': 'bl2_32_a.BLI/bl3.DEBLA'}), (
        'blabli/blebla[foo]', {'name': 'blabli/blebla', 'sub_part': 'foo'}),
        ('blabli/blebla[/foo/bar]', {'name': 'blabli/blebla', 'sub_part':
        '/foo/bar'}), ('blabli/blebla[foo]', {'name': 'blabli/blebla',
        'sub_part': 'foo'}), ('C:\\A\\L0cAL\\Windows\\file.ext', {'name':
        'C:\\A\\L0cAL\\Windows\\file.ext', 'sub_part': None, 'version':
        None}), ('/modelkit/tmp-local-asset/1.0/subpart/README.md', {'name':
        '/modelkit/tmp-local-asset/1.0/subpart/README.md', 'sub_part': None,
        'version': None})]
    for version in versions:
        examples += [(f'blabli/blebla:{version}[/foo/bar]', {'name':
            'blabli/blebla', 'sub_part': '/foo/bar', 'version': version}),
            (f'blabli/blebla:{version}[/foo]', {'name': 'blabli/blebla',
            'sub_part': '/foo', 'version': version}), (
            f'blabli/BLEBLA:{version}[FOO]', {'name': 'blabli/BLEBLA',
            'sub_part': 'FOO', 'version': version})]
    return examples
