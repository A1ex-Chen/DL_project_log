def initialize_parameters(default_model='unoMT_default_model.txt'):
    unoMTb = unoMT.unoMTBk(unoMT.file_path, default_model, 'pytorch', prog=
        'unoMT_baseline', desc=
        'Multi-task combined single and combo drug prediction for cross-study data - Pilot 1'
        )
    print('Created unoMT benchmark')
    gParameters = candle.finalize_parameters(unoMTb)
    print('Parameters initialized')
    return gParameters
