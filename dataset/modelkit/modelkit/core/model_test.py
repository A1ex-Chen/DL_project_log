def test(self):
    console = Console()
    for i, (model_key, item, expected, keyword_args) in enumerate(self.
        _iterate_test_cases(model_key=self.configuration_key)):
        result = None
        try:
            if isinstance(self, AsyncModel):
                result = AsyncToSync(self.predict)(item, **keyword_args)
            else:
                result = self.predict(item, **keyword_args)
            assert result == expected
            console.print(f'[green]TEST {i + 1}: SUCCESS[/green]')
        except AssertionError:
            console.print('[red]TEST {}: FAILED[/red]{} test failed on item'
                .format(i + 1, ' [' + model_key + ']' if model_key else ''))
            t = Tree('item')
            console.print(describe(item, t=t))
            t = Tree('expected')
            console.print(describe(expected, t=t))
            t = Tree('result')
            console.print(describe(result, t=t))
            raise
