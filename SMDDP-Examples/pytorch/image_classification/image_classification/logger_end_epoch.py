def end_epoch(self):
    for n, m in self.metrics.items():
        if not n.startswith('calib'):
            m['meter'].reset_iteration()
    verbositys = {m['level'] for _, m in self.metrics.items()}
    for ll in verbositys:
        llm = {n: m for n, m in self.metrics.items() if m['level'] == ll}
        dllogger.log(step=(self.epoch,), data={n: m['meter'].get_epoch() for
            n, m in llm.items()})
