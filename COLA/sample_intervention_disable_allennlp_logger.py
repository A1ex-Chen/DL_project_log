def disable_allennlp_logger():
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('transformers.generation_utils').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(
        logging.INFO)
