def train_model(gParameters, models, X_train, Y_train, X_test, Y_test, fold,
    verbose=False):
    base_run_id = gParameters['run_id']
    for epoch in range(gParameters['epochs']):
        for k in range(len(models)):
            model = models[k]
            gParameters['run_id'] = base_run_id + '.{}.{}.{}'.format(fold,
                epoch, k)
            candleRemoteMonitor = candle.CandleRemoteMonitor(params=gParameters
                )
            timeoutMonitor = candle.TerminateOnTimeOut(gParameters['timeout'])
            model.fit({'input': X_train[k]}, {('out_' + str(k)): Y_train[k]
                }, epochs=1, verbose=verbose, callbacks=[
                candleRemoteMonitor, timeoutMonitor], batch_size=
                gParameters['batch_size'], validation_data=(X_test[k],
                Y_test[k]))
    return models
