def main():
    gParameters = initialize_parameters()
    benchmark.check_params(gParameters)
    run(gParameters)
