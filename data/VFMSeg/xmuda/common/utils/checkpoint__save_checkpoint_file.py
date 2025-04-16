def _save_checkpoint_file(self, path):
    with open(path, 'w') as f:
        lines = []
        for p in self._last_checkpoints:
            if not os.path.isabs(p):
                p = os.path.basename(p)
            lines.append(p)
        f.write('\n'.join(lines))
