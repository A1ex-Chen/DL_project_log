def find_cuda_device_arch():
    if sys.platform == 'win32':
        return None
    cuda_home = find_cuda()
    if cuda_home is None:
        return None
    cuda_home = Path(cuda_home)
    try:
        device_query_path = cuda_home / 'extras/demo_suite/deviceQuery'
        if not device_query_path.exists():
            source = """
            #include <cuda_runtime.h>
            #include <iostream>
            int main(){
                int nDevices;
                cudaGetDeviceCount(&nDevices);
                for (int i = 0; i < nDevices; i++) {
                    cudaDeviceProp prop;
                    cudaGetDeviceProperties(&prop, i);
                    std::cout << prop.major << "." << prop.minor << std::endl;
                }
                return 0;
            }
            """
            with tempfile.NamedTemporaryFile('w', suffix='.cc') as f:
                f_path = Path(f.name)
                f.write(source)
                f.flush()
                try:
                    cmd = (
                        f"g++ {f.name} -o {f_path.stem} -I{cuda_home / 'include'} -L{cuda_home / 'lib64'} -lcudart"
                        )
                    print(cmd)
                    subprocess.check_output(cmd, shell=True, cwd=f_path.parent)
                    cmd = f'./{f_path.stem}'
                    arches = subprocess.check_output(cmd, shell=True, cwd=
                        f_path.parent).decode().rstrip('\r\n').split('\n')
                    if len(arches) < 1:
                        return None
                    arch = arches[0]
                except:
                    return None
        else:
            cmd = f"{str(device_query_path)} | grep 'CUDA Capability'"
            arch = subprocess.check_output(cmd, shell=True).decode().rstrip(
                '\r\n').split(' ')[-1]
        arch = f'sm_{arch[0]}{arch[-1]}'
    except Exception:
        arch = None
    return arch
