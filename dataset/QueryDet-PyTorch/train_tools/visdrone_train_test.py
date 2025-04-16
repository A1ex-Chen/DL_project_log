@classmethod
def test(cls, cfg, model, evaluators=None):
    logger = logging.getLogger(__name__)
    dataset_name = 'VisDrone2018'
    data_loader = cls.build_test_loader(cfg, dataset_name)
    evaluator = cls.build_evaluator(cfg, dataset_name)
    result = inference_on_dataset(model, data_loader, evaluator)
    return []
