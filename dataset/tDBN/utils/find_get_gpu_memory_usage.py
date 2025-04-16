def get_gpu_memory_usage():
    if sys.platform == 'win32':
        return None
    cuda_home = find_cuda()
    if cuda_home is None:
        return None
    cuda_home = Path(cuda_home)
    source = """
    #include <cuda_runtime.h>
    #include <iostream>
    int main(){
        int nDevices;
        cudaGetDeviceCount(&nDevices);
        size_t free_m, total_m;
        // output json format.
        std::cout << "[";
        for (int i = 0; i < nDevices; i++) {
            cudaSetDevice(i);
            cudaMemGetInfo(&free_m, &total_m);
            std::cout << "[" << free_m << "," << total_m << "]";
            if (i != nDevices - 1)
                std::cout << "," << std::endl;
        }
        std::cout << "]" << std::endl;
        return 0;
    }
    """
    with tempfile.NamedTemporaryFile('w', suffix='.cc') as f:
        f_path = Path(f.name)
        f.write(source)
        f.flush()
        try:
            cmd = (
                f"g++ {f.name} -o {f_path.stem} -std=c++11 -I{cuda_home / 'include'} -L{cuda_home / 'lib64'} -lcudart"
                )
            print(cmd)
            subprocess.check_output(cmd, shell=True, cwd=f_path.parent)
            cmd = f'./{f_path.stem}'
            usages = subprocess.check_output(cmd, shell=True, cwd=f_path.parent
                ).decode()
            usages = json.loads(usages)
            return usages
        except:
            return None
    return None
