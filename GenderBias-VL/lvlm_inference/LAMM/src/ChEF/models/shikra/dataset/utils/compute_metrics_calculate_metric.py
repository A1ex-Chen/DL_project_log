def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]
    ) ->Dict[str, Any]:
    correct = 0
    failed = 0
    target_failed = 0
    for pred, target in zip(preds, targets):
        extract_pred = self.extract_ans(pred)
        extract_target = self.extract_ans(target)
        if extract_target is None:
            target_failed += 1
            logger.warning(
                f'failed to extract ans from target. maybe the response string is truncated: {target}.'
                )
            continue
        if extract_pred is None:
            failed += 1
        if extract_pred == extract_target:
            correct += 1
    return {'accuracy': 1.0 * correct / len(targets), 'target_failed':
        target_failed, 'failed': failed}
