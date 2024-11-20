def get_data(gParams):
    data_url = gParams['data_url']
    gParams['vqvae'] = candle.fetch_file(data_url + gParams['vqvae'],
        subdir='Examples/histogen')
    gParams['top'] = candle.fetch_file(data_url + gParams['top'], subdir=
        'Examples/histogen')
    gParams['bottom'] = candle.fetch_file(data_url + gParams['bottom'],
        subdir='Examples/histogen')
