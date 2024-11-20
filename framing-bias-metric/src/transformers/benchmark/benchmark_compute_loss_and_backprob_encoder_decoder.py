def compute_loss_and_backprob_encoder_decoder():
    loss = train_model(input_ids, decoder_input_ids=input_ids, labels=input_ids
        )[0]
    loss.backward()
    return loss
