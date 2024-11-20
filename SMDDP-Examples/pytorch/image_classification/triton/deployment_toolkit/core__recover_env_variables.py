def _recover_env_variables(self, old_envs: Dict[str, object]):
    for name, value in old_envs.items():
        if value is None:
            del os.environ[name]
        else:
            os.environ[name] = str(value)
