@classmethod
def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference')
    evaluator_list = []
    evaluator_list.append(JsonEvaluator(os.path.join(cfg.OUTPUT_DIR,
        'visdrone_infer.json'), class_add_1=True))
    if cfg.META_INFO.EVAL_GPU_TIME:
        evaluator_list.append(GPUTimeEvaluator(True, 'minisecond'))
    return DatasetEvaluators(evaluator_list)
