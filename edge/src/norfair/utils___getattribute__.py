def __getattribute__(self, name):
    print(
        """[bold red]Missing dependency:[/bold red] You are trying to use Norfair's metrics features without the required dependencies.

Please, install Norfair with `pip install norfair\\[metrics]`, or `pip install norfair\\[metrics,video]` if you also want video features."""
        )
    exit()
