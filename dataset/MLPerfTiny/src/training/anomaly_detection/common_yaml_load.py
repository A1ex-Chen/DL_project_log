def yaml_load():
    with open('baseline.yaml') as stream:
        param = yaml.safe_load(stream)
    return param
