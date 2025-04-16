def compute_loss_and_backprob_encoder():
    loss = train_model(input_ids, labels=input_ids)[0]
    loss.backward()
    return loss
