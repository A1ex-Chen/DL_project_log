def tensorrt_official_qat():
    try:
        quant_modules.initialize()
    except NameError:
        logging.info('initialzation error for quant_modules')
