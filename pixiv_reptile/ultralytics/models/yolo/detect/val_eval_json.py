def eval_json(self, stats):
    """Evaluates YOLO output in JSON format and returns performance statistics."""
    if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.
        jdict):
        pred_json = self.save_dir / 'predictions.json'
        anno_json = self.data['path'] / 'annotations' / (
            'instances_val2017.json' if self.is_coco else
            f'lvis_v1_{self.args.split}.json')
        pkg = 'pycocotools' if self.is_coco else 'lvis'
        LOGGER.info(
            f'\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...')
        try:
            for x in (pred_json, anno_json):
                assert x.is_file(), f'{x} file not found'
            check_requirements('pycocotools>=2.0.6' if self.is_coco else
                'lvis>=0.5.3')
            if self.is_coco:
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval
                anno = COCO(str(anno_json))
                pred = anno.loadRes(str(pred_json))
                val = COCOeval(anno, pred, 'bbox')
            else:
                from lvis import LVIS, LVISEval
                anno = LVIS(str(anno_json))
                pred = anno._load_json(str(pred_json))
                val = LVISEval(anno, pred, 'bbox')
            val.params.imgIds = [int(Path(x).stem) for x in self.dataloader
                .dataset.im_files]
            val.evaluate()
            val.accumulate()
            val.summarize()
            if self.is_lvis:
                val.print_results()
            stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]
                ] = val.stats[:2] if self.is_coco else [val.results['AP50'],
                val.results['AP']]
        except Exception as e:
            LOGGER.warning(f'{pkg} unable to run: {e}')
    return stats
