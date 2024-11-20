@classmethod
def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference')
    evaluator_list = []
    if cfg.META_INFO.EVAL_AP:
        evaluator_list.append(COCOEvaluatorFPN(dataset_name, cfg, True,
            output_folder))
    return DatasetEvaluators(evaluator_list)
