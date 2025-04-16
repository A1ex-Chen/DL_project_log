def datagen(self, epoch=0, print_out=1, test=0):
    files = self.files
    if not test:
        order = random.sample(list(self.train_ind), int(self.
            sampling_density * len(self.train_ind)))
    else:
        order = self.test_ind
    for f_ind in order:
        if print_out:
            print(files[f_ind], '\n')
        X, nbrs, resnums = helper.get_data_arrays(files[f_ind])
        if self.type_feature:
            Xnorm = np.concatenate([X[:, :, :, 0:3] / 320.0, X[:, :, :, 3:8
                ], X[:, :, :, 8:] / 10.0], axis=3)
        else:
            Xnorm = np.concatenate([X[:, :, :, 0:3] / 320.0, X[:, :, :, 8:] /
                10.0], axis=3)
        num_frames = X.shape[0]
        xt_all = np.array([])
        yt_all = np.array([])
        num_active_frames = random.sample(range(num_frames), int(self.
            sampling_density * num_frames))
        print('Datagen on the following frames', num_active_frames)
        for i in num_active_frames:
            if self.conv_net:
                xt = Xnorm[i]
                if self.nbr_type == 'relative':
                    xt = helper.append_nbrs_relative(xt, nbrs[i], self.
                        molecular_nbrs)
                elif self.nbr_type == 'invariant':
                    xt = helper.append_nbrs_invariant(xt, nbrs[i], self.
                        molecular_nbrs)
                else:
                    print('Invalid nbr_type')
                    exit()
                yt = xt.copy()
                xt = xt.reshape(xt.shape[0], 1, xt.shape[1], 1)
                if self.full_conv_net:
                    yt = xt.copy()
            else:
                xt = Xnorm[i]
                if self.nbr_type == 'relative':
                    xt = helper.append_nbrs_relative(xt, nbrs[i], self.
                        molecular_nbrs)
                elif self.nbr_type == 'invariant':
                    xt = helper.append_nbrs_invariant(xt, nbrs[i], self.
                        molecular_nbrs)
                else:
                    print('Invalid nbr_type')
                    exit()
                yt = xt.copy()
            if not len(xt_all):
                xt_all = np.expand_dims(xt, axis=0)
                yt_all = np.expand_dims(yt, axis=0)
            else:
                xt_all = np.append(xt_all, np.expand_dims(xt, axis=0), axis=0)
                yt_all = np.append(yt_all, np.expand_dims(yt, axis=0), axis=0)
        yield files[f_ind], xt_all, yt_all
    return
