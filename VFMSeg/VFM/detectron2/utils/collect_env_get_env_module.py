def get_env_module():
    var_name = 'DETECTRON2_ENV_MODULE'
    return var_name, os.environ.get(var_name, '<not set>')
