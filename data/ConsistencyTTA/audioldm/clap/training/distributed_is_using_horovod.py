def is_using_horovod():
    ompi_vars = ['OMPI_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_SIZE']
    pmi_vars = ['PMI_RANK', 'PMI_SIZE']
    if all([(var in os.environ) for var in ompi_vars]) or all([(var in os.
        environ) for var in pmi_vars]):
        return True
    else:
        return False
