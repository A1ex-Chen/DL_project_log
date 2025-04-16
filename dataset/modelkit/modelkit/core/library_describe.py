def describe(self, console=None) ->None:
    if not console:
        console = Console()
    t = Tree('[bold]Settings')
    console.print(describe(self.settings, t=t))
    t = Tree('[bold]Configuration')
    console.print(describe(self.configuration, t=t))
    t = Tree('[bold]Assets')
    if not self.assets_info:
        t.add('[dim][italic]No assets loaded')
    else:
        describe(self.assets_info, t=t)
    console.print(t)
    t = Tree('[bold]Models')
    if not self.models:
        t.add('[dim][italic]No models loaded')
    else:
        describe(self.models, t=t)
    console.print(t)
