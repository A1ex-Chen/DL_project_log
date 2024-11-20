def run(params):
    candle.set_seed(params['rng_seed'])
    uqmode = params['uqmode']
    filename = params['results_filename']
    cv = params['cv']
    index_dp = filename.find('DR=')
    if index_dp == -1:
        print('No dropout rate found in filename')
        print('Using -1 to denote NA')
        dp_perc = -1
    else:
        if filename[index_dp + 6] == '.':
            dp = float(filename[index_dp + 3:index_dp + 3 + 3])
        else:
            dp = float(filename[index_dp + 3:index_dp + 3 + 4])
        print('Droput rate: ', dp)
        dp_perc = dp * 100.0
    method = uqmode + ' - dropout ' + str(dp_perc) + '%'
    prefix = params['output_dir'] + '/' + uqmode + '_DR=' + str(dp_perc)
    df_data = pd.read_csv(filename, sep='\t')
    print('data read shape: ', df_data.shape)
    if uqmode == 'hom':
        if df_data.shape[1] < 9:
            print(
                'Too few columns... Asumming that a summary  (and not individual realizations) has been  given as input'
                )
            Ytest, Ypred_mean, yerror, sigma, Ypred_std, pred_name = (candle
                .compute_statistics_homoscedastic_summary(df_data))
        else:
            Ytest, Ypred_mean, yerror, sigma, Ypred_std, pred_name = (candle
                .compute_statistics_homoscedastic(df_data))
        cov80p = coverage_80p(Ytest, Ypred_mean, sigma)
    elif uqmode == 'het':
        Ytest, Ypred_mean, yerror, sigma, Ypred_std, pred_name = (candle.
            compute_statistics_heteroscedastic(df_data))
        cov80p = coverage_80p(Ytest, Ypred_mean, sigma)
    elif uqmode == 'qtl':
        (Ytest, Ypred_mean, yerror, sigma, Ypred_std, pred_name,
            Ypred_1d_mean, Ypred_9d_mean) = candle.compute_statistics_quantile(
            df_data)
        cov80p = coverage_80p(Ytest, Ypred_mean, None, Ypred_1d_mean,
            Ypred_9d_mean)
        decile_list = ['5th', '1st', '9th']
        candle.plot_decile_predictions(Ypred_mean, Ypred_1d_mean,
            Ypred_9d_mean, decile_list, pred_name, prefix)
    elif uqmode == 'contam':
        Ytest, Ypred_mean, yerror, sigma_, Ypred_std, pred_name = (candle.
            compute_statistics_homoscedastic(df_data))
        sigma_scalar = params['sigma']
        if sigma_scalar is None:
            raise Exception(
                'ERROR ! No sigma specified for contamination model... Exiting'
                )
        sigma = sigma_scalar * np.ones(Ytest.shape[0])
        cov80p = coverage_80p(Ytest, Ypred_mean, sigma)
        print('Coverage (80%): ', cov80p)
        candle.plot_density_observed_vs_predicted(Ytest, Ypred_mean,
            pred_name, prefix)
        candle.plot_2d_density_sigma_vs_error(sigma, yerror, method, prefix)
        mse = np.mean((Ytest - Ypred_mean) ** 2)
        mae = np.mean(np.abs(Ytest - Ypred_mean))
        print('Prediction error in testing')
        print('MSE: ', mse)
        print('MAE: ', mae)
        candle.plot_contamination(Ytest, Ypred_mean, sigma, pred_name=
            pred_name, figprefix=prefix)
        print(
            'Since in contamination model std prediction is uniform for all samples, no point in calibrating... Finishing'
            )
        return
    else:
        raise Exception('ERROR ! UQ mode specified for calibration: ' +
            uqmode + ' not implemented... Exiting')
    print('Coverage (80%) before calibration: ', cov80p)
    candle.plot_density_observed_vs_predicted(Ytest, Ypred_mean, pred_name,
        prefix)
    candle.plot_2d_density_sigma_vs_error(sigma, yerror, method, prefix)
    candle.plot_histogram_error_per_sigma(sigma, yerror, method, prefix)
    (index_perm_total, pSigma_cal, pSigma_test, pMean_cal, pMean_test,
        true_cal, true_test) = (candle.split_data_for_empirical_calibration
        (Ytest, Ypred_mean, sigma))
    splineobj1, splineobj2 = (candle.
        compute_empirical_calibration_interpolation(pSigma_cal, pMean_cal,
        true_cal, cv))
    error = np.abs(true_cal - pMean_cal)
    candle.plot_calibration_interpolation(pSigma_cal, error, splineobj1,
        splineobj2, method, prefix, params['plot_steps'])
    eabs_pred = splineobj2(pSigma_test)
    cov80p = coverage_80p(true_test, pMean_test, eabs_pred)
    print('Coverage (80%) after calibration: ', cov80p)
    eabs_true = np.abs(true_test - pMean_test)
    mse = np.mean((eabs_true - eabs_pred) ** 2)
    mae = np.mean(np.abs(eabs_true - eabs_pred))
    print('Prediction error in testing calibration')
    print('MSE: ', mse)
    print('MAE: ', mae)
    candle.plot_calibrated_std(true_test, pMean_test, eabs_pred, 2.0 * mae,
        pred_name, prefix)
    fname = prefix + '_calibration_interpolation_spline.dkl'
    with open(fname, 'wb') as f:
        dill.dump(splineobj2, f)
        print('Calibration spline (interpolation) stored in file: ', fname)
