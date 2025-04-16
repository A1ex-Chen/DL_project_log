def retrieve_images(self, retrieved_images, prompt_embeds, knn=10):
    if self.retriever is not None:
        additional_images = self.retriever.retrieve_imgs_batch(prompt_embeds
            [:, 0].cpu(), knn).total_examples
        for i in range(len(retrieved_images)):
            retrieved_images[i] += additional_images[i][self.retriever.
                config.image_column]
    return retrieved_images
