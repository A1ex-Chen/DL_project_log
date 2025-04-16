def export_fmts_hub():
    """Returns a list of HUB-supported export formats."""
    from ultralytics.engine.exporter import export_formats
    return list(export_formats()['Argument'][1:]) + ['ultralytics_tflite',
        'ultralytics_coreml']
