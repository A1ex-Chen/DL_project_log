def dataframe(self):
    results = self.acc
    results['loss'] = self.loss
    return pd.DataFrame(results)
