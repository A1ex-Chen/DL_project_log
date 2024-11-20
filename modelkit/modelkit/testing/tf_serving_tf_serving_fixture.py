def tf_serving_fixture(request, lib, tf_version, deployment='docker'):
    cmd = ['--port=8500', '--rest_api_port=8501']
    if deployment == 'process':
        deploy_tf_models(lib, 'local-process', config_name='testing')
        proc = subprocess.Popen(['tensorflow_model_server',
            f"--model_config_file={os.environ['MODELKIT_ASSETS_DIR']}/testing.config"
            ] + cmd)

        def finalize():
            proc.terminate()
    else:
        deploy_tf_models(lib, 'local-docker', config_name='testing')
        subprocess.call(['docker', 'rm', '-f', 'modelkit-tfserving-tests'],
            stderr=subprocess.DEVNULL)
        tfserving_proc = subprocess.Popen(['docker', 'run', '--name',
            'modelkit-tfserving-tests', '--volume',
            f"{os.environ['MODELKIT_ASSETS_DIR']}:/config", '-p',
            '8500:8500', '-p', '8501:8501',
            f'tensorflow/serving:{tf_version}',
            '--model_config_file=/config/testing.config'] + cmd)

        def finalize():
            subprocess.call(['docker', 'kill', 'modelkit-tfserving-tests'])
            tfserving_proc.terminate()
    request.addfinalizer(finalize)
    connect_tf_serving(next(x for x in lib.required_models if issubclass(
        lib.configuration[x].model_type, TensorflowModel)), 'localhost', 
        8500, 'grpc')
