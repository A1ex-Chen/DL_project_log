@torch.no_grad()
def retrieve(self, model, preprocess, elements, search_text: str, device
    ) ->int:
    """Processes images and text with a model, calculates similarity, and returns softmax score."""
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = self.clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100.0 * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)
