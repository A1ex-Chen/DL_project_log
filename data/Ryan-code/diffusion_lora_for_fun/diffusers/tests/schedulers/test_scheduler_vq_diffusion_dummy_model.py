def dummy_model(self, num_vec_classes):

    def model(sample, t, *args):
        batch_size, num_latent_pixels = sample.shape
        logits = torch.rand((batch_size, num_vec_classes - 1,
            num_latent_pixels))
        return_value = F.log_softmax(logits.double(), dim=1).float()
        return return_value
    return model
