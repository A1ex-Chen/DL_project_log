def _get_module(self, module_name: str):
    try:
        return importlib.import_module('.' + module_name, self.__name__)
    except Exception as e:
        raise RuntimeError(
            f"""Failed to import {self.__name__}.{module_name} because of the following error (look up to see its traceback):
{e}"""
            ) from e
