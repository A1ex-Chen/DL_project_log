def frame(self):
    return pd.DataFrame(self.data).set_index('epoch_index')
