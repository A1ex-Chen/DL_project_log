def eval_json(self, stats):
    """Evaluates YOLO output in JSON format and returns performance statistics."""
    if self.args.save_json and self.is_dota and len(self.jdict):
        import json
        import re
        from collections import defaultdict
        pred_json = self.save_dir / 'predictions.json'
        pred_txt = self.save_dir / 'predictions_txt'
        pred_txt.mkdir(parents=True, exist_ok=True)
        data = json.load(open(pred_json))
        LOGGER.info(f'Saving predictions with DOTA format to {pred_txt}...')
        for d in data:
            image_id = d['image_id']
            score = d['score']
            classname = self.names[d['category_id']].replace(' ', '-')
            p = d['poly']
            with open(f"{pred_txt / f'Task1_{classname}'}.txt", 'a') as f:
                f.writelines(
                    f"""{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}
"""
                    )
        pred_merged_txt = self.save_dir / 'predictions_merged_txt'
        pred_merged_txt.mkdir(parents=True, exist_ok=True)
        merged_results = defaultdict(list)
        LOGGER.info(
            f'Saving merged predictions with DOTA format to {pred_merged_txt}...'
            )
        for d in data:
            image_id = d['image_id'].split('__')[0]
            pattern = re.compile('\\d+___\\d+')
            x, y = (int(c) for c in re.findall(pattern, d['image_id'])[0].
                split('___'))
            bbox, score, cls = d['rbox'], d['score'], d['category_id']
            bbox[0] += x
            bbox[1] += y
            bbox.extend([score, cls])
            merged_results[image_id].append(bbox)
        for image_id, bbox in merged_results.items():
            bbox = torch.tensor(bbox)
            max_wh = torch.max(bbox[:, :2]).item() * 2
            c = bbox[:, 6:7] * max_wh
            scores = bbox[:, 5]
            b = bbox[:, :5].clone()
            b[:, :2] += c
            i = ops.nms_rotated(b, scores, 0.3)
            bbox = bbox[i]
            b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
            for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                classname = self.names[int(x[-1])].replace(' ', '-')
                p = [round(i, 3) for i in x[:-2]]
                score = round(x[-2], 3)
                with open(f"{pred_merged_txt / f'Task1_{classname}'}.txt", 'a'
                    ) as f:
                    f.writelines(
                        f"""{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}
"""
                        )
    return stats
