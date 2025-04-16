def export_tfjs(file, prefix=colorstr('TensorFlow.js:')):
    try:
        check_requirements(('tensorflowjs',))
        import re
        import tensorflowjs as tfjs
        LOGGER.info(
            f'\n{prefix} starting export with tensorflowjs {tfjs.__version__}...'
            )
        f = str(file).replace('.pt', '_web_model')
        f_pb = file.with_suffix('.pb')
        f_json = f'{f}/model.json'
        cmd = (
            f'tensorflowjs_converter --input_format=tf_frozen_model --output_node_names=Identity,Identity_1,Identity_2,Identity_3 {f_pb} {f}'
            )
        subprocess.run(cmd.split())
        with open(f_json) as j:
            json = j.read()
        with open(f_json, 'w') as j:
            subst = re.sub(
                '{"outputs": {"Identity.?.?": {"name": "Identity.?.?"}, "Identity.?.?": {"name": "Identity.?.?"}, "Identity.?.?": {"name": "Identity.?.?"}, "Identity.?.?": {"name": "Identity.?.?"}}}'
                ,
                '{"outputs": {"Identity": {"name": "Identity"}, "Identity_1": {"name": "Identity_1"}, "Identity_2": {"name": "Identity_2"}, "Identity_3": {"name": "Identity_3"}}}'
                , json)
            j.write(subst)
        LOGGER.info(
            f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')
