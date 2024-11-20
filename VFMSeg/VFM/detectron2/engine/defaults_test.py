@classmethod
def test(cls, cfg, model, evaluators=None):
    """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
    logger = logging.getLogger(__name__)
    if isinstance(evaluators, DatasetEvaluator):
        evaluators = [evaluators]
    if evaluators is not None:
        assert len(cfg.DATASETS.TEST) == len(evaluators), '{} != {}'.format(len
            (cfg.DATASETS.TEST), len(evaluators))
    results = OrderedDict()
    for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
        data_loader = cls.build_test_loader(cfg, dataset_name)
        if evaluators is not None:
            evaluator = evaluators[idx]
        else:
            try:
                evaluator = cls.build_evaluator(cfg, dataset_name)
            except NotImplementedError:
                logger.warn(
                    'No evaluator found. Use `DefaultTrainer.test(evaluators=)`, or implement its `build_evaluator` method.'
                    )
                results[dataset_name] = {}
                continue
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            assert isinstance(results_i, dict
                ), 'Evaluator must return a dict on the main process. Got {} instead.'.format(
                results_i)
            logger.info('Evaluation results for {} in csv format:'.format(
                dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results
