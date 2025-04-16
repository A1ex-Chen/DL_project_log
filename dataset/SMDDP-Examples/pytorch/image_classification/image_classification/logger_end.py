def end(self):
    for n, m in self.metrics.items():
        m['meter'].reset_epoch()
    verbositys = {m['level'] for _, m in self.metrics.items()}
    for ll in verbositys:
        llm = {n: m for n, m in self.metrics.items() if m['level'] == ll}
        dllogger.log(step=tuple(), data={n: m['meter'].get_run() for n, m in
            llm.items()})
    for n, m in self.metrics.items():
        m['meter'].reset_epoch()
    dllogger.flush()
