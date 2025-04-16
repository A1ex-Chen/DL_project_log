def main():
    python_files = glob.glob(PATTERN)
    for file in python_files:
        print(f'****** Running file: {file} ******')
        if file != 'benchmark_text_to_image.py':
            command = f'python {file}'
            run_command(command.split())
            command += ' --run_compile'
            run_command(command.split())
    for file in python_files:
        if file == 'benchmark_text_to_image.py':
            for ckpt in ALL_T2I_CKPTS:
                command = f'python {file} --ckpt {ckpt}'
                if 'turbo' in ckpt:
                    command += ' --num_inference_steps 1'
                run_command(command.split())
                command += ' --run_compile'
                run_command(command.split())
        elif file == 'benchmark_sd_img.py':
            for ckpt in ['stabilityai/stable-diffusion-xl-refiner-1.0',
                'stabilityai/sdxl-turbo']:
                command = f'python {file} --ckpt {ckpt}'
                if ckpt == 'stabilityai/sdxl-turbo':
                    command += ' --num_inference_steps 2'
                run_command(command.split())
                command += ' --run_compile'
                run_command(command.split())
        elif file in ['benchmark_sd_inpainting.py', 'benchmark_ip_adapters.py'
            ]:
            sdxl_ckpt = 'stabilityai/stable-diffusion-xl-base-1.0'
            command = f'python {file} --ckpt {sdxl_ckpt}'
            run_command(command.split())
            command += ' --run_compile'
            run_command(command.split())
        elif file in ['benchmark_controlnet.py', 'benchmark_t2i_adapter.py']:
            sdxl_ckpt = ('diffusers/controlnet-canny-sdxl-1.0' if 
                'controlnet' in file else
                'TencentARC/t2i-adapter-canny-sdxl-1.0')
            command = f'python {file} --ckpt {sdxl_ckpt}'
            run_command(command.split())
            command += ' --run_compile'
            run_command(command.split())
