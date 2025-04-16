def check_avail(self):
    if MultiTensorApply.available == False:
        raise RuntimeError(
            'Attempted to call MultiTensorApply method, but MultiTensorApply is not available, possibly because Apex was installed without --cpp_ext --cuda_ext.  Original import error message:'
            , MultiTensorApply.import_err)
