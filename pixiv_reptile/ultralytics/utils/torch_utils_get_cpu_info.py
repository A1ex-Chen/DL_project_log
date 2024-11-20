def get_cpu_info():
    """Return a string with system CPU information, i.e. 'Apple M2'."""
    import cpuinfo
    k = 'brand_raw', 'hardware_raw', 'arch_string_raw'
    info = cpuinfo.get_cpu_info()
    string = info.get(k[0] if k[0] in info else k[1] if k[1] in info else k
        [2], 'unknown')
    return string.replace('(R)', '').replace('CPU ', '').replace('@ ', '')
