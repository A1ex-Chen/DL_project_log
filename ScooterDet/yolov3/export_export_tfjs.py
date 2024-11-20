@try_export
def export_tfjs(file, int8, prefix=colorstr('TensorFlow.js:')):
    check_requirements('tensorflowjs')
    import tensorflowjs as tfjs
    LOGGER.info(
        f'\n{prefix} starting export with tensorflowjs {tfjs.__version__}...')
    f = str(file).replace('.pt', '_web_model')
    f_pb = file.with_suffix('.pb')
    f_json = f'{f}/model.json'
    args = ['tensorflowjs_converter', '--input_format=tf_frozen_model', 
        '--quantize_uint8' if int8 else '',
        '--output_node_names=Identity,Identity_1,Identity_2,Identity_3',
        str(f_pb), f]
    subprocess.run([arg for arg in args if arg], check=True)
    json = Path(f_json).read_text()
    with open(f_json, 'w') as j:
        subst = re.sub(
            '{"outputs": {"Identity.?.?": {"name": "Identity.?.?"}, "Identity.?.?": {"name": "Identity.?.?"}, "Identity.?.?": {"name": "Identity.?.?"}, "Identity.?.?": {"name": "Identity.?.?"}}}'
            ,
            '{"outputs": {"Identity": {"name": "Identity"}, "Identity_1": {"name": "Identity_1"}, "Identity_2": {"name": "Identity_2"}, "Identity_3": {"name": "Identity_3"}}}'
            , json)
        j.write(subst)
    return f, None
