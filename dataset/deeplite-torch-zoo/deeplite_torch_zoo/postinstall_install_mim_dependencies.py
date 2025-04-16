def install_mim_dependencies():
    mim_dep = ['mmpretrain[mim]>=1.0.0rc8', 'mmyolo[mim]==0.6.0']
    print('Checking zoo dependencies, please wait...')
    import mim
    mim.install(mim_dep)
    print('Check over')
