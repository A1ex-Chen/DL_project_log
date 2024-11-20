def load_combined_dose_response(rename=True):
    df1 = load_single_dose_response(combo_format=True)
    logger.info('Loaded {} single drug dose response measurements'.format(
        df1.shape[0]))
    df2 = load_combo_dose_response()
    logger.info('Loaded {} drug pair dose response measurements'.format(df2
        .shape[0]))
    df = pd.concat([df1, df2])
    logger.info('Combined dose response data contains sources: {}'.format(
        df['SOURCE'].unique()))
    if rename:
        df = df.rename(columns={'SOURCE': 'Source', 'CELL': 'Sample',
            'DRUG1': 'Drug1', 'DRUG2': 'Drug2', 'DOSE1': 'Dose1', 'DOSE2':
            'Dose2', 'GROWTH': 'Growth', 'STUDY': 'Study'})
    return df
