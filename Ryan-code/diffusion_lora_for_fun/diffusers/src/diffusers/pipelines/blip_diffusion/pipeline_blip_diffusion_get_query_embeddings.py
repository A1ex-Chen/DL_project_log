def get_query_embeddings(self, input_image, src_subject):
    return self.qformer(image_input=input_image, text_input=src_subject,
        return_dict=False)
