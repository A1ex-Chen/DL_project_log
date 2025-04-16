def loss_plot(self, loss, val_loss):
    """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
    ax = self.fig.add_subplot(1, 1, 1)
    ax.cla()
    ax.plot(loss)
    ax.plot(val_loss)
    ax.set_title('Model loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(['Train', 'Validation'], loc='upper right')
