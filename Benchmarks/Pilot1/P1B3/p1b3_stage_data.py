def stage_data():
    server = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B3/'
    cell_expr_path = candle.fetch_file(server +
        'P1B3_cellline_expressions.tsv', 'Pilot1', unpack=False)
    cell_mrna_path = candle.fetch_file(server + 'P1B3_cellline_mirna.tsv',
        'Pilot1', unpack=False)
    cell_prot_path = candle.fetch_file(server +
        'P1B3_cellline_proteome.tsv', 'Pilot1', unpack=False)
    cell_kino_path = candle.fetch_file(server + 'P1B3_cellline_kinome.tsv',
        'Pilot1', unpack=False)
    drug_desc_path = candle.fetch_file(server + 'P1B3_drug_descriptors.tsv',
        'Pilot1', unpack=False)
    drug_auen_path = candle.fetch_file(server + 'P1B3_drug_latent.csv',
        'Pilot1', unpack=False)
    dose_resp_path = candle.fetch_file(server + 'P1B3_dose_response.csv',
        'Pilot1', unpack=False)
    test_cell_path = candle.fetch_file(server + 'P1B3_test_celllines.txt',
        'Pilot1', unpack=False)
    test_drug_path = candle.fetch_file(server + 'P1B3_test_drugs.txt',
        'Pilot1', unpack=False)
    return (cell_expr_path, cell_mrna_path, cell_prot_path, cell_kino_path,
        drug_desc_path, drug_auen_path, dose_resp_path, test_cell_path,
        test_drug_path)
