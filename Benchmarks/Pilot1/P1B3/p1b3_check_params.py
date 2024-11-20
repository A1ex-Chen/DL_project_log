def check_params(fileParams):
    try:
        fileParams['dense']
    except KeyError:
        try:
            fileParams['conv']
        except KeyError:
            print(
                'Error! No dense or conv layers specified. Wrong file !! ... exiting '
                )
            raise
        else:
            try:
                fileParams['pool']
            except KeyError:
                fileParams['pool'] = None
                print('Warning ! No pooling specified after conv layer.')
