def train_ac(self):
    for i in range(1, self.mb_epochs + 1):
        print('\nTraining epoch: {:d}\n'.format(i))
        frame_loss = []
        frame_mse = []
        current_path = self.save_path + 'epoch_' + str(i)
        if not os.path.exists(current_path):
            os.makedirs(self.save_path + '/epoch_' + str(i))
        model_weight_file = '%s/%s.hdf5' % (current_path, 'model_weights')
        encoder_weight_file = '%s/%s.hdf5' % (current_path, 'encoder_weights')
        for curr_file, xt_all, yt_all in self.datagen(i):
            for frame in range(len(xt_all)):
                history = self.molecular_model.fit(xt_all[frame], yt_all[
                    frame], epochs=1, batch_size=self.batch_size, callbacks
                    =self.callbacks[:2])
                frame_loss.append(history.history['loss'])
                frame_mse.append(history.history['mean_squared_error'])
                if not frame % 20 or self.sampling_density != 1.0:
                    self.molecular_model.save_weights(model_weight_file)
                    self.molecular_encoder.save_weights(encoder_weight_file)
        print('Saving loss and mse after current epoch... \n')
        np.save(current_path + '/loss.npy', frame_loss)
        np.save(current_path + '/mse.npy', frame_mse)
        print('Saving weights after current epoch... \n')
        self.molecular_model.save_weights(model_weight_file)
        self.molecular_encoder.save_weights(encoder_weight_file)
        print('Saving latent space output for current epoch... \n')
        for curr_file, xt_all, yt_all in self.datagen(0, 0, test=1):
            XP = []
            for frame in range(len(xt_all)):
                yp = self.molecular_encoder.predict(xt_all[frame],
                    batch_size=self.batch_size)
                XP.append(yp)
            XP = np.array(XP)
            fout = (current_path + '/' + curr_file.split('/')[-1].split(
                '.npz')[0] + '_AE' + '_Include%s' % self.type_feature + 
                '_Conv%s' % self.conv_net + '.npy')
            print(fout)
            np.save(fout, XP)
    return frame_loss, frame_mse
