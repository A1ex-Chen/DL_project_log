def eval_json(self, stats):
    """Return COCO-style object detection evaluation metrics."""
    if self.args.save_json and self.is_coco and len(self.jdict):
        anno_json = self.data['path'] / 'annotations/instances_val2017.json'
        pred_json = self.save_dir / 'predictions.json'
        LOGGER.info(
            f'\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...'
            )
        try:
            check_requirements('pycocotools>=2.0.6')
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            for x in (anno_json, pred_json):
                assert x.is_file(), f'{x} file not found'
            anno = COCO(str(anno_json))
            pred = anno.loadRes(str(pred_json))
            for i, eval in enumerate([COCOeval(anno, pred, 'bbox'),
                COCOeval(anno, pred, 'segm')]):
                if self.is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in self.
                        dataloader.dataset.im_files]
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                idx = i * 4 + 2
                stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]
                    ] = eval.stats[:2]
        except Exception as e:
            LOGGER.warning(f'pycocotools unable to run: {e}')
    return stats
