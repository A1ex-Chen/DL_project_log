# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
"""
import os
import argparse
import subprocess
import time
import warnings

import cv2
import matplotlib.pyplot as plt
import mock    # pip install mock
import numpy as np
import torch

from src.args import ArgumentParserRGBDSegmentation

from src.models.model_utils import SqueezeAndExcitationTensorRT
from src.datasets.sunrgbd.sunrgbd import SUNRBDBase
from src.prepare_data import prepare_data

with mock.patch('src.models.model_utils.SqueezeAndExcitation',
                SqueezeAndExcitationTensorRT):
    from src.build_model import build_model
















if __name__ == '__main__':
    args = _parse_args()
    print(f"args: {vars(args)}")

    print('PyTorch version:', torch.__version__)

    if args.time_tensorrt:
        import tensorrt as trt
        import pycuda.autoinit
        import pycuda.driver as cuda

        print('TensorRT version:', trt.__version__)

    if args.time_onnxruntime:
        import onnxruntime

        onnxruntime_profile_execution = True

        # see: https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/TensorRT-ExecutionProvider.md
        os.environ['ORT_TENSORRT_MAX_WORKSPACE_SIZE'] = str(2 << 30)
        os.environ['ORT_TENSORRT_MIN_SUBGRAPH_SIZE'] = '1'  # 5
        # note, 1 does not raise an error if not available but enabled
        os.environ['ORT_TENSORRT_FP16_ENABLE'] = '0'   # 1
        os.environ['ORT_TENSORRT_MAX_PARTITION_ITERATIONS'] = \
            args.onnxruntime_trt_max_partition_iterations

        print('ONNXRuntime version:', onnxruntime.__version__)
        print('ONNXRuntime available providers:',
              onnxruntime.get_available_providers())

    gpu_devices = torch.cuda.device_count()

    # prepare inputs ----------------------------------------------------------
    label_downsampling_rates = []
    results_dir = os.path.join(os.path.dirname(__file__),
                               f'inference_results_{args.upsampling}',
                               args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    args.batch_size = 1
    args.batch_size_valid = 1

    rgb_images = []
    depth_images = []
    if args.dataset_dir is not None:
        # get samples from dataset
        _, valid_loader, *additional = prepare_data(args)
        if args.valid_full_res:
            # use full res valid loader
            valid_loader = additional[0]
        dataset = valid_loader.dataset

        for i, sample in enumerate(valid_loader):
            if i == (args.n_runs + args.n_runs_warmup):
                break
            rgb_images.append(sample['image'])
            depth_images.append(sample['depth'])
    else:
        # get random samples
        dataset, preprocessor = prepare_data(args)

        for _ in range(args.n_runs + args.n_runs_warmup):
            img_rgb = np.random.randint(0, 255,
                                        size=(args.height, args.width, 3),
                                        dtype='uint8')
            img_depth = np.random.randint(0, 40000,
                                          size=(args.height, args.width),
                                          dtype='uint16')
            # preprocess
            sample = preprocessor({'image': img_rgb, 'depth': img_depth})
            rgb_images.append(sample['image'][None])
            depth_images.append(sample['depth'][None])

    n_classes_without_void = dataset.n_classes_without_void

    if args.modality == 'rgbd':
        inputs = (rgb_images, depth_images)
    elif args.modality == 'rgb':
        inputs = (rgb_images,)
    elif args.modality == 'depth':
        inputs = (depth_images,)
    else:
        raise NotImplementedError()

    # create model ------------------------------------------------------------
    if args.model is 'onnx' and args.time_pytorch:
        warnings.warn("PyTorch inference timing disabled since onnx model is "
                      "given")
        args.time_pytorch = False

    if args.model == 'own':
        model, device = build_model(args, n_classes_without_void)

        # load weights
        if args.last_ckpt:
            checkpoint = torch.load(args.last_ckpt,
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'], strict=True)

        model.eval()
        model.to(device)
    else:
        # onnx model given
        assert args.model_onnx_filepath is not None

    # time inference using PyTorch --------------------------------------------
    if args.time_pytorch:
        timings_pytorch, outs_pytorch = time_inference_pytorch(
            model,
            inputs,
            device,
            n_runs_warmup=args.n_runs_warmup
        )
        print(f'fps pytorch: {np.mean(1/timings_pytorch):0.4f} ± '
              f'{np.std(1/timings_pytorch):0.4f}')

    # time inference using TensorRT -------------------------------------------
    if args.time_tensorrt:
        if args.model_onnx_filepath is None:
            dummy_inputs = [inp[0].to(device) for inp in inputs]

            input_names = [f'input_{i}' for i in range(len(dummy_inputs))]
            output_names = ['output']
            onnx_filepath = './model_tensorrt.onnx'

            torch.onnx.export(model,
                              tuple(dummy_inputs),
                              onnx_filepath,
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              do_constant_folding=True,
                              verbose=False,
                              opset_version=args.trt_onnx_opset_version)
            print(f"ONNX file written to '{onnx_filepath}'.")
        else:
            onnx_filepath = args.model_onnx_filepath

        timings_tensorrt, outs_tensorrt = time_inference_tensorrt(
            onnx_filepath,
            inputs,
            trt_floatx=args.trt_floatx,
            trt_batchsize=args.trt_batchsize,
            trt_workspace=args.trt_workspace,
            n_runs_warmup=args.n_runs_warmup,
            force_tensorrt_engine_rebuild=args.trt_force_rebuild,
        )

        print(f'fps tensorrt: {np.mean(1/timings_tensorrt):0.4f} ± '
              f'{np.std(1/timings_tensorrt):0.4f}')

    # time inference using ONNXRuntime ----------------------------------------
    if args.time_onnxruntime:
        if args.model_onnx_filepath is None:
            dummy_inputs = [inp[0].to(device) for inp in inputs]

            input_names = [f'input_{i}' for i in range(len(dummy_inputs))]
            output_names = ['output']
            onnx_filepath = './model_onnxruntime.onnx'

            torch.onnx.export(
                model,
                tuple(dummy_inputs),
                onnx_filepath,
                export_params=True,
                input_names=input_names,
                output_names=output_names,
                do_constant_folding=True,
                verbose=False,
                opset_version=args.onnxruntime_onnx_opset_version
            )
            print(f"ONNX file written to '{onnx_filepath}'.\n")
            input("Press [ENTER] to continue interfence timing in the same "
                  "run or [CTRL+C] to stop here and rerun the script with "
                  "--model_onnx_filepath to lower memory consumption.")
        else:
            onnx_filepath = args.model_onnx_filepath

        timings_onnxruntime, outs_onnxruntime = time_inference_onnxruntime(
            onnx_filepath,
            inputs,
            n_runs_warmup=args.n_runs_warmup,
            profile_execution=onnxruntime_profile_execution
        )

        print(f'fps onnxruntime: {np.mean(1/timings_onnxruntime):0.4f} ± '
              f'{np.std(1/timings_onnxruntime):0.4f}')

    # plot/export results -----------------------------------------------------
    if args.plot_timing:
        plt.figure()
        if 'timings_pytorch' in locals():
            plt.plot(1 / timings_pytorch, label='pytorch')
        if 'timings_tensorrt' in locals():
            plt.plot(1 / timings_tensorrt, label='tensorrt')
        if 'timings_onnxruntime' in locals():
            plt.plot(1 / timings_onnxruntime, label='onnxruntime')
        plt.xlabel("run")
        plt.ylabel("fps")
        plt.legend()
        plt.title("Inference time")
        plt.show()

    if args.plot_outputs or args.export_outputs:
        if 'timings_pytorch' in locals():
            for i, out_pytorch in enumerate(outs_pytorch):
                argmax_pytorch = np.argmax(out_pytorch.numpy()[0],
                                           axis=0).astype(np.uint8) + 1
                colored = dataset.color_label(argmax_pytorch)

                if args.export_outputs:
                    save_path = os.path.join(results_dir,
                                             f'{i:04d}_jetson_pytorch.png')
                    save_path_colored = os.path.join(
                        results_dir, f'{i:04d}_jetson_pytorch_colored.png')

                    cv2.imwrite(save_path, argmax_pytorch)
                    cv2.imwrite(save_path_colored,
                                cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

                if args.plot_outputs:
                    plt.figure()
                    plt.imshow(colored)
                    plt.title("Pytorch")
                    plt.show()

        if 'timings_tensorrt' in locals():
            for i, out_tensorrt in enumerate(outs_tensorrt):
                out = out_tensorrt.reshape(-1, args.height, args.width)

                argmax_tensorrt = np.argmax(out, axis=0).astype(np.uint8) + 1
                colored = dataset.color_label(argmax_tensorrt)

                if args.export_outputs:
                    save_path = os.path.join(
                        results_dir,
                        f'{i:04d}_jetson_tensorrt_float{args.trt_floatx}.png'
                    )
                    save_path_colored = os.path.join(
                        results_dir,
                        f'{i:04d}_jetson_tensorrt_float{args.trt_floatx}'
                        f'_colored.png'
                    )

                    cv2.imwrite(save_path, argmax_tensorrt)
                    cv2.imwrite(save_path_colored,
                                cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

                if args.plot_outputs:
                    plt.figure()
                    plt.imshow(colored)
                    plt.title("TensorRT")
                    plt.show()

        if 'timings_onnxruntime' in locals():
            if os.environ['ORT_TENSORRT_FP16_ENABLE'] == '1':
                floatx = '16'
            else:
                floatx = '32'
            for i, out_onnxruntime in enumerate(outs_onnxruntime):
                out = out_onnxruntime.reshape(-1, args.height, args.width)

                argmax_onnxruntime = np.argmax(out,
                                               axis=0).astype(np.uint8) + 1
                colored = dataset.color_label(argmax_onnxruntime)

                if args.export_outputs:
                    save_path = os.path.join(
                        results_dir,
                        f'{i:04d}_jetson_onnxruntime_float{floatx}.png')
                    save_path_colored = os.path.join(
                        results_dir,
                        f'{i:04d}_jetson_onnxruntime_float{floatx}'
                        f'_colored.png')

                    cv2.imwrite(save_path, argmax_onnxruntime)
                    cv2.imwrite(save_path_colored,
                                cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

                if args.plot_outputs:
                    plt.figure()
                    plt.imshow(colored)
                    plt.title("ONNXRuntime")
                    plt.show()