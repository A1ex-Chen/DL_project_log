def gather_predictions(predictions_list, detokenize):
    rnnt_predictions = (__rnnt_decoder_predictions_tensor(prediction,
        detokenize) for prediction in predictions_list)
    return [prediction for batch in rnnt_predictions for prediction in batch]
