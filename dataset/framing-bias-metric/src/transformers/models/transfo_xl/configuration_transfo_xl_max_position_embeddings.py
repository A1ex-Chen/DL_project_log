@property
def max_position_embeddings(self):
    logger.info(
        f'The model {self.model_type} is one of the few models that has no sequence length limit.'
        )
    return -1
