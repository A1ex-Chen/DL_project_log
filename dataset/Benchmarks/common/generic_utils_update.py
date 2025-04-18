def update(self, current, values=[], force=False):
    """
        Parameters
        ------------
        current : int
            index of current step
        values : list of tuples (name, value_for_last_step).
            The progress bar will display averages for these values.
        force : boolean
            force visual progress update
        """
    for k, v in values:
        if k not in self.sum_values:
            self.sum_values[k] = [v * (current - self.seen_so_far), current -
                self.seen_so_far]
            self.unique_values.append(k)
        else:
            self.sum_values[k][0] += v * (current - self.seen_so_far)
            self.sum_values[k][1] += current - self.seen_so_far
    self.seen_so_far = current
    now = time.time()
    if self.verbose == 1:
        if not force and now - self.last_update < self.interval:
            return
        prev_total_width = self.total_width
        sys.stdout.write('\x08' * prev_total_width)
        sys.stdout.write('\r')
        numdigits = int(np.floor(np.log10(self.target))) + 1
        barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
        bar = barstr % (current, self.target)
        prog = float(current) / self.target
        prog_width = int(self.width * prog)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.target:
                bar += '>'
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        sys.stdout.write(bar)
        self.total_width = len(bar)
        if current:
            time_per_unit = (now - self.start) / current
        else:
            time_per_unit = 0
        eta = time_per_unit * (self.target - current)
        info = ''
        if current < self.target:
            info += ' - ETA: %ds' % eta
        else:
            info += ' - %ds' % (now - self.start)
        for k in self.unique_values:
            info += ' - %s:' % k
            if type(self.sum_values[k]) is list:
                avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                if abs(avg) > 0.001:
                    info += ' %.4f' % avg
                else:
                    info += ' %.4e' % avg
            else:
                info += ' %s' % self.sum_values[k]
        self.total_width += len(info)
        if prev_total_width > self.total_width:
            info += (prev_total_width - self.total_width) * ' '
        sys.stdout.write(info)
        sys.stdout.flush()
        if current >= self.target:
            sys.stdout.write('\n')
    if self.verbose == 2:
        if current >= self.target:
            info = '%ds' % (now - self.start)
            for k in self.unique_values:
                info += ' - %s:' % k
                avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                if avg > 0.001:
                    info += ' %.4f' % avg
                else:
                    info += ' %.4e' % avg
            sys.stdout.write(info + '\n')
    self.last_update = now
