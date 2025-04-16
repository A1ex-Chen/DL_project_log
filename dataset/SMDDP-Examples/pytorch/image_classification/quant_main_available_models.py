def available_models():
    models = {m.name: m for m in [efficientnet_quant_b0, efficientnet_quant_b4]
        }
    return models
