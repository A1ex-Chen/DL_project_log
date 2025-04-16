def print_results(self):
    print('NSB results (K={}{}):'.format(self.number_of_bins, 
        ', data whitening' if self.whitening else ''))
    for model in sorted(list(self.cached_results.keys())):
        res = self.cached_results[model]
        print('%s: NDB = %d, NDB/K = %.3f, JS = %.4f' % (model, res['NDB'],
            res['NDB'] / self.number_of_bins, res['JS']))
