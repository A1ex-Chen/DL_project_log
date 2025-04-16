@classmethod
def test(cls, cfg, model, evaluators=None):
    logger = logging.getLogger(__name__)
    dataset_name = 'VisDrone2018'
    data_loader = cls.build_test_loader(cfg, dataset_name)
    evaluator = cls.build_evaluator(cfg, dataset_name)
    result = inference_on_dataset(model, data_loader, evaluator)
    if comm.is_main_process():
        assert isinstance(result, dict
            ), 'Evaluator must return a dict on the main process. Got {} instead.'.format(
            result)
        logger.info('Evaluation results for {} in csv format:'.format(
            dataset_name))
        print_csv_format(result)
    if len(result) == 1:
        result = list(result.values())[0]
    return result
