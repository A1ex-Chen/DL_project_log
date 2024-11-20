def main():
    params = initialize_parameters()
    if params['infer'] is True:
        run_inference(params)
    else:
        run(params)
