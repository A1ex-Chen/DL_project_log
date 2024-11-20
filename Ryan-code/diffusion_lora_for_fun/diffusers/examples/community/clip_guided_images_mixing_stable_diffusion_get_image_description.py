def get_image_description(self, image):
    transformed_image = self.coca_transform(image).unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = self.coca_model.generate(transformed_image.to(device=
            self.device, dtype=self.coca_model.dtype))
    generated = self.coca_tokenizer.decode(generated[0].cpu().numpy())
    return generated.split('<end_of_text>')[0].replace('<start_of_text>', ''
        ).rstrip(' .,')
