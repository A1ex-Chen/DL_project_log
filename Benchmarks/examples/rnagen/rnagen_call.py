def call(self, data):
    x, y = data
    z_mean, z_log_var, z = self.encoder(x)
    if type(x) == tuple:
        y_pred = self.decoder([z, x[1]])
    else:
        y_pred = self.decoder(z)
    return y_pred
