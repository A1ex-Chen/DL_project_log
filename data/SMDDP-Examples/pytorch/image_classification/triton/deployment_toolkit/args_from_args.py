def from_args(self, args: Union[argparse.Namespace, Dict]):
    args = self.get_args(args)
    LOGGER.info(f'Initializing {self._cls_or_fn.__name__}({args})')
    return self._cls_or_fn(**args)
