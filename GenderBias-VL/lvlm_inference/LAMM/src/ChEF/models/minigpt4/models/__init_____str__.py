def __str__(self) ->str:
    return ('=' * 50 + '\n' + f"{'Architectures':<30} {'Types'}\n" + '=' * 
        50 + '\n' + '\n'.join([f"{name:<30} {', '.join(types)}" for name,
        types in self.model_zoo.items()]))
