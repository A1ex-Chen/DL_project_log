import argparse
import sys

import tensorrt as trt




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sd_xl", action="store_true", default=False, help="SD XL pipeline")

    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Path to the onnx checkpoint to convert",
    )

    parser.add_argument("--num_controlnet", type=int)

    parser.add_argument("--output_path", type=str, required=True, help="Path to the output model.")

    parser.add_argument("--fp16", action="store_true", default=False, help="Export the models in `float16` mode")

    args = parser.parse_args()

    convert_models(args.onnx_path, args.num_controlnet, args.output_path, args.fp16, args.sd_xl)