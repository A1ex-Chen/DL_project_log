def get_model(params):
    url = params['model_url']
    file_model = ('DIR.ml.' + params['base_name'] +
        '.Orderable_zinc_db_enaHLL.sorted.4col.dd.parquet/' +
        'reg_go.autosave.model.h5')
    model_file = candle.get_file(file_model, url + file_model, cache_subdir
        ='Pilot1')
    return model_file
