def check_torchvision():
    """
    Checks the installed versions of PyTorch and Torchvision to ensure they're compatible.

    This function checks the installed versions of PyTorch and Torchvision, and warns if they're incompatible according
    to the provided compatibility table based on:
    https://github.com/pytorch/vision#installation.

    The compatibility table is a dictionary where the keys are PyTorch versions and the values are lists of compatible
    Torchvision versions.
    """
    compatibility_table = {'2.3': ['0.18'], '2.2': ['0.17'], '2.1': ['0.16'
        ], '2.0': ['0.15'], '1.13': ['0.14'], '1.12': ['0.13']}
    v_torch = '.'.join(torch.__version__.split('+')[0].split('.')[:2])
    if v_torch in compatibility_table:
        compatible_versions = compatibility_table[v_torch]
        v_torchvision = '.'.join(TORCHVISION_VERSION.split('+')[0].split(
            '.')[:2])
        if all(v_torchvision != v for v in compatible_versions):
            print(
                f"""WARNING ⚠️ torchvision=={v_torchvision} is incompatible with torch=={v_torch}.
Run 'pip install torchvision=={compatible_versions[0]}' to fix torchvision or 'pip install -U torch torchvision' to update both.
For a full compatibility table see https://github.com/pytorch/vision#installation"""
                )
