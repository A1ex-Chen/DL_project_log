def get_pixelcnn_prior_loss(x, output):
    q, logit_probs = output
    return nn.CrossEntropyLoss()(logit_probs, q)
