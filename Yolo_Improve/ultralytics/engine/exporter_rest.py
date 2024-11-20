# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Export a YOLOv8 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolov8n.pt
TorchScript             | `torchscript`             | yolov8n.torchscript
ONNX                    | `onnx`                    | yolov8n.onnx
OpenVINO                | `openvino`                | yolov8n_openvino_model/
TensorRT                | `engine`                  | yolov8n.engine
CoreML                  | `coreml`                  | yolov8n.mlpackage
TensorFlow SavedModel   | `saved_model`             | yolov8n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolov8n.pb
TensorFlow Lite         | `tflite`                  | yolov8n.tflite
TensorFlow Edge TPU     | `edgetpu`                 | yolov8n_edgetpu.tflite
TensorFlow.js           | `tfjs`                    | yolov8n_web_model/
PaddlePaddle            | `paddle`                  | yolov8n_paddle_model/
NCNN                    | `ncnn`                    | yolov8n_ncnn_model/

Requirements:
    $ pip install "ultralytics[export]"

Python:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    results = model.export(format='onnx')

CLI:
    $ yolo mode=export model=yolov8n.pt format=onnx

Inference:
    $ yolo predict model=yolov8n.pt                 # PyTorch
                         yolov8n.torchscript        # TorchScript
                         yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                         yolov8n_openvino_model     # OpenVINO
                         yolov8n.engine             # TensorRT
                         yolov8n.mlpackage          # CoreML (macOS-only)
                         yolov8n_saved_model        # TensorFlow SavedModel
                         yolov8n.pb                 # TensorFlow GraphDef
                         yolov8n.tflite             # TensorFlow Lite
                         yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                         yolov8n_paddle_model       # PaddlePaddle
                         yolov8n_ncnn_model         # NCNN

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov8n_web_model public/yolov8n_web_model
    $ npm start
"""

import gc
import json
import os
import shutil
import subprocess
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ultralytics.cfg import TASK2DATA, get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import check_class_names, default_class_names
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder
from ultralytics.nn.tasks import DetectionModel, SegmentationModel, WorldModel
from ultralytics.utils import (
    ARM64,
    DEFAULT_CFG,
    IS_JETSON,
    LINUX,
    LOGGER,
    MACOS,
    PYTHON_VERSION,
    ROOT,
    WINDOWS,
    __version__,
    callbacks,
    colorstr,
    get_default_args,
    yaml_save,
)
from ultralytics.utils.checks import check_imgsz, check_is_path_safe, check_requirements, check_version
from ultralytics.utils.downloads import attempt_download_asset, get_github_assets, safe_download
from ultralytics.utils.files import file_size, spaces_in_path
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import TORCH_1_13, get_latest_opset, select_device, smart_inference_mode







    return outer_func


class Exporter:
    """
    A class for exporting a model.

    Attributes:
        args (SimpleNamespace): Configuration for the exporter.
        callbacks (list, optional): List of callback functions. Defaults to None.
    """


    @smart_inference_mode()


    @try_export

    @try_export

    @try_export

    @try_export

    @try_export

    @try_export

    @try_export

    @try_export

    @try_export

    @try_export

    @try_export

    @try_export






class IOSDetectModel(torch.nn.Module):
    """Wrap an Ultralytics YOLO model for Apple iOS CoreML export."""



        if self.args.int8:
            fq = str(self.file).replace(self.file.suffix, f"_int8_openvino_model{os.sep}")
            fq_ov = str(Path(fq) / self.file.with_suffix(".xml").name)
            check_requirements("nncf>=2.8.0")
            import nncf


            # Generate calibration data for integer quantization
            ignored_scope = None
            if isinstance(self.model.model[-1], Detect):
                # Includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
                head_module_name = ".".join(list(self.model.named_modules())[-1][0].split(".")[:2])
                ignored_scope = nncf.IgnoredScope(  # ignore operations
                    patterns=[
                        f".*{head_module_name}/.*/Add",
                        f".*{head_module_name}/.*/Sub*",
                        f".*{head_module_name}/.*/Mul*",
                        f".*{head_module_name}/.*/Div*",
                        f".*{head_module_name}\\.dfl.*",
                    ],
                    types=["Sigmoid"],
                )

            quantized_ov_model = nncf.quantize(
                model=ov_model,
                calibration_dataset=nncf.Dataset(self.get_int8_calibration_dataloader(prefix), transform_fn),
                preset=nncf.QuantizationPreset.MIXED,
                ignored_scope=ignored_scope,
            )
            serialize(quantized_ov_model, fq_ov)
            return fq, None

        f = str(self.file).replace(self.file.suffix, f"_openvino_model{os.sep}")
        f_ov = str(Path(f) / self.file.with_suffix(".xml").name)

        serialize(ov_model, f_ov)
        return f, None

    @try_export
    def export_paddle(self, prefix=colorstr("PaddlePaddle:")):
        """YOLOv8 Paddle export."""
        check_requirements(("paddlepaddle", "x2paddle"))
        import x2paddle  # noqa
        from x2paddle.convert import pytorch2paddle  # noqa

        LOGGER.info(f"\n{prefix} starting export with X2Paddle {x2paddle.__version__}...")
        f = str(self.file).replace(self.file.suffix, f"_paddle_model{os.sep}")

        pytorch2paddle(module=self.model, save_dir=f, jit_type="trace", input_examples=[self.im])  # export
        yaml_save(Path(f) / "metadata.yaml", self.metadata)  # add metadata.yaml
        return f, None

    @try_export
    def export_ncnn(self, prefix=colorstr("NCNN:")):
        """
        YOLOv8 NCNN export using PNNX https://github.com/pnnx/pnnx.
        """
        check_requirements("ncnn")
        import ncnn  # noqa

        LOGGER.info(f"\n{prefix} starting export with NCNN {ncnn.__version__}...")
        f = Path(str(self.file).replace(self.file.suffix, f"_ncnn_model{os.sep}"))
        f_ts = self.file.with_suffix(".torchscript")

        name = Path("pnnx.exe" if WINDOWS else "pnnx")  # PNNX filename
        pnnx = name if name.is_file() else (ROOT / name)
        if not pnnx.is_file():
            LOGGER.warning(
                f"{prefix} WARNING ⚠️ PNNX not found. Attempting to download binary file from "
                "https://github.com/pnnx/pnnx/.\nNote PNNX Binary file must be placed in current working directory "
                f"or in {ROOT}. See PNNX repo for full installation instructions."
            )
            system = "macos" if MACOS else "windows" if WINDOWS else "linux-aarch64" if ARM64 else "linux"
            try:
                release, assets = get_github_assets(repo="pnnx/pnnx")
                asset = [x for x in assets if f"{system}.zip" in x][0]
                assert isinstance(asset, str), "Unable to retrieve PNNX repo assets"  # i.e. pnnx-20240410-macos.zip
                LOGGER.info(f"{prefix} successfully found latest PNNX asset file {asset}")
            except Exception as e:
                release = "20240410"
                asset = f"pnnx-{release}-{system}.zip"
                LOGGER.warning(f"{prefix} WARNING ⚠️ PNNX GitHub assets not found: {e}, using default {asset}")
            unzip_dir = safe_download(f"https://github.com/pnnx/pnnx/releases/download/{release}/{asset}", delete=True)
            if check_is_path_safe(Path.cwd(), unzip_dir):  # avoid path traversal security vulnerability
                (unzip_dir / name).rename(pnnx)  # move binary to ROOT
                pnnx.chmod(0o777)  # set read, write, and execute permissions for everyone
                shutil.rmtree(unzip_dir)  # delete unzip dir

        ncnn_args = [
            f'ncnnparam={f / "model.ncnn.param"}',
            f'ncnnbin={f / "model.ncnn.bin"}',
            f'ncnnpy={f / "model_ncnn.py"}',
        ]

        pnnx_args = [
            f'pnnxparam={f / "model.pnnx.param"}',
            f'pnnxbin={f / "model.pnnx.bin"}',
            f'pnnxpy={f / "model_pnnx.py"}',
            f'pnnxonnx={f / "model.pnnx.onnx"}',
        ]

        cmd = [
            str(pnnx),
            str(f_ts),
            *ncnn_args,
            *pnnx_args,
            f"fp16={int(self.args.half)}",
            f"device={self.device.type}",
            f'inputshape="{[self.args.batch, 3, *self.imgsz]}"',
        ]
        f.mkdir(exist_ok=True)  # make ncnn_model directory
        LOGGER.info(f"{prefix} running '{' '.join(cmd)}'")
        subprocess.run(cmd, check=True)

        # Remove debug files
        pnnx_files = [x.split("=")[-1] for x in pnnx_args]
        for f_debug in ("debug.bin", "debug.param", "debug2.bin", "debug2.param", *pnnx_files):
            Path(f_debug).unlink(missing_ok=True)

        yaml_save(f / "metadata.yaml", self.metadata)  # add metadata.yaml
        return str(f), None

    @try_export
    def export_coreml(self, prefix=colorstr("CoreML:")):
        """YOLOv8 CoreML export."""
        mlmodel = self.args.format.lower() == "mlmodel"  # legacy *.mlmodel export format requested
        check_requirements("coremltools>=6.0,<=6.2" if mlmodel else "coremltools>=7.0")
        import coremltools as ct  # noqa

        LOGGER.info(f"\n{prefix} starting export with coremltools {ct.__version__}...")
        assert not WINDOWS, "CoreML export is not supported on Windows, please run on macOS or Linux."
        assert self.args.batch == 1, "CoreML batch sizes > 1 are not supported. Please retry at 'batch=1'."
        f = self.file.with_suffix(".mlmodel" if mlmodel else ".mlpackage")
        if f.is_dir():
            shutil.rmtree(f)

        bias = [0.0, 0.0, 0.0]
        scale = 1 / 255
        classifier_config = None
        if self.model.task == "classify":
            classifier_config = ct.ClassifierConfig(list(self.model.names.values())) if self.args.nms else None
            model = self.model
        elif self.model.task == "detect":
            model = IOSDetectModel(self.model, self.im) if self.args.nms else self.model
        else:
            if self.args.nms:
                LOGGER.warning(f"{prefix} WARNING ⚠️ 'nms=True' is only available for Detect models like 'yolov8n.pt'.")
                # TODO CoreML Segment and Pose model pipelining
            model = self.model

        ts = torch.jit.trace(model.eval(), self.im, strict=False)  # TorchScript model
        ct_model = ct.convert(
            ts,
            inputs=[ct.ImageType("image", shape=self.im.shape, scale=scale, bias=bias)],
            classifier_config=classifier_config,
            convert_to="neuralnetwork" if mlmodel else "mlprogram",
        )
        bits, mode = (8, "kmeans") if self.args.int8 else (16, "linear") if self.args.half else (32, None)
        if bits < 32:
            if "kmeans" in mode:
                check_requirements("scikit-learn")  # scikit-learn package required for k-means quantization
            if mlmodel:
                ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
            elif bits == 8:  # mlprogram already quantized to FP16
                import coremltools.optimize.coreml as cto

                op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=bits, weight_threshold=512)
                config = cto.OptimizationConfig(global_config=op_config)
                ct_model = cto.palettize_weights(ct_model, config=config)
        if self.args.nms and self.model.task == "detect":
            if mlmodel:
                # coremltools<=6.2 NMS export requires Python<3.11
                check_version(PYTHON_VERSION, "<3.11", name="Python ", hard=True)
                weights_dir = None
            else:
                ct_model.save(str(f))  # save otherwise weights_dir does not exist
                weights_dir = str(f / "Data/com.apple.CoreML/weights")
            ct_model = self._pipeline_coreml(ct_model, weights_dir=weights_dir)

        m = self.metadata  # metadata dict
        ct_model.short_description = m.pop("description")
        ct_model.author = m.pop("author")
        ct_model.license = m.pop("license")
        ct_model.version = m.pop("version")
        ct_model.user_defined_metadata.update({k: str(v) for k, v in m.items()})
        try:
            ct_model.save(str(f))  # save *.mlpackage
        except Exception as e:
            LOGGER.warning(
                f"{prefix} WARNING ⚠️ CoreML export to *.mlpackage failed ({e}), reverting to *.mlmodel export. "
                f"Known coremltools Python 3.11 and Windows bugs https://github.com/apple/coremltools/issues/1928."
            )
            f = f.with_suffix(".mlmodel")
            ct_model.save(str(f))
        return f, ct_model

    @try_export
    def export_engine(self, prefix=colorstr("TensorRT:")):
        """YOLOv8 TensorRT export https://developer.nvidia.com/tensorrt."""
        assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        # self.args.simplify = True
        f_onnx, _ = self.export_onnx()  # run before TRT import https://github.com/ultralytics/ultralytics/issues/7016

        try:
            import tensorrt as trt  # noqa
        except ImportError:
            if LINUX:
                check_requirements("tensorrt>7.0.0,<=10.1.0")
            import tensorrt as trt  # noqa
        check_version(trt.__version__, ">=7.0.0", hard=True)
        check_version(trt.__version__, "<=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")

        # Setup and checks
        LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
        is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # is TensorRT >= 10
        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
        f = self.file.with_suffix(".engine")  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if self.args.verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        # Engine builder
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        workspace = int(self.args.workspace * (1 << 30))
        if is_trt10:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        else:  # TensorRT versions 7, 8
            config.max_workspace_size = workspace
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        half = builder.platform_has_fast_fp16 and self.args.half
        int8 = builder.platform_has_fast_int8 and self.args.int8
        # Read ONNX file
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(f_onnx):
            raise RuntimeError(f"failed to load ONNX file: {f_onnx}")

        # Network inputs
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

        if self.args.dynamic:
            shape = self.im.shape
            if shape[0] <= 1:
                LOGGER.warning(f"{prefix} WARNING ⚠️ 'dynamic=True' model requires max batch size, i.e. 'batch=16'")
            profile = builder.create_optimization_profile()
            min_shape = (1, shape[1], 32, 32)  # minimum input shape
            max_shape = (*shape[:2], *(max(1, self.args.workspace) * d for d in shape[2:]))  # max input shape
            for inp in inputs:
                profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
            config.add_optimization_profile(profile)

        LOGGER.info(f"{prefix} building {'INT8' if int8 else 'FP' + ('16' if half else '32')} engine as {f}")
        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_calibration_profile(profile)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

            class EngineCalibrator(trt.IInt8Calibrator):






            # Load dataset w/ builder (for batching) and calibrate
            config.int8_calibrator = EngineCalibrator(
                dataset=self.get_int8_calibration_dataloader(prefix),
                batch=2 * self.args.batch,
                cache=str(self.file.with_suffix(".cache")),
            )

        elif half:
            config.set_flag(trt.BuilderFlag.FP16)

        # Free CUDA memory
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        # Write file
        build = builder.build_serialized_network if is_trt10 else builder.build_engine
        with build(network, config) as engine, open(f, "wb") as t:
            # Metadata
            meta = json.dumps(self.metadata)
            t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
            t.write(meta.encode())
            # Model
            t.write(engine if is_trt10 else engine.serialize())

        return f, None

    @try_export
    def export_saved_model(self, prefix=colorstr("TensorFlow SavedModel:")):
        """YOLOv8 TensorFlow SavedModel export."""
        cuda = torch.cuda.is_available()
        try:
            import tensorflow as tf  # noqa
        except ImportError:
            suffix = "-macos" if MACOS else "-aarch64" if ARM64 else "" if cuda else "-cpu"
            version = ">=2.0.0"
            check_requirements(f"tensorflow{suffix}{version}")
            import tensorflow as tf  # noqa
        check_requirements(
            (
                "keras",  # required by 'onnx2tf' package
                "tf_keras",  # required by 'onnx2tf' package
                "sng4onnx>=1.0.1",  # required by 'onnx2tf' package
                "onnx_graphsurgeon>=0.3.26",  # required by 'onnx2tf' package
                "onnx>=1.12.0",
                "onnx2tf>1.17.5,<=1.22.3",
                "onnxslim>=0.1.31",
                "tflite_support<=0.4.3" if IS_JETSON else "tflite_support",  # fix ImportError 'GLIBCXX_3.4.29'
                "flatbuffers>=23.5.26,<100",  # update old 'flatbuffers' included inside tensorflow package
                "onnxruntime-gpu" if cuda else "onnxruntime",
            ),
            cmds="--extra-index-url https://pypi.ngc.nvidia.com",  # onnx_graphsurgeon only on NVIDIA
        )

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
        check_version(
            tf.__version__,
            ">=2.0.0",
            name="tensorflow",
            verbose=True,
            msg="https://github.com/ultralytics/ultralytics/issues/5161",
        )
        import onnx2tf

        f = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        if f.is_dir():
            shutil.rmtree(f)  # delete output folder

        # Pre-download calibration file to fix https://github.com/PINTO0309/onnx2tf/issues/545
        onnx2tf_file = Path("calibration_image_sample_data_20x128x128x3_float32.npy")
        if not onnx2tf_file.exists():
            attempt_download_asset(f"{onnx2tf_file}.zip", unzip=True, delete=True)

        # Export to ONNX
        self.args.simplify = True
        f_onnx, _ = self.export_onnx()

        # Export to TF
        np_data = None
        if self.args.int8:
            tmp_file = f / "tmp_tflite_int8_calibration_images.npy"  # int8 calibration images file
            verbosity = "info"
            if self.args.data:
                f.mkdir()
                images = [batch["img"].permute(0, 2, 3, 1) for batch in self.get_int8_calibration_dataloader(prefix)]
                images = torch.cat(images, 0).float()
                # mean = images.view(-1, 3).mean(0)  # imagenet mean [123.675, 116.28, 103.53]
                # std = images.view(-1, 3).std(0)  # imagenet std [58.395, 57.12, 57.375]
                np.save(str(tmp_file), images.numpy().astype(np.float32))  # BHWC
                np_data = [["images", tmp_file, [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]]
        else:
            verbosity = "error"

        LOGGER.info(f"{prefix} starting TFLite export with onnx2tf {onnx2tf.__version__}...")
        onnx2tf.convert(
            input_onnx_file_path=f_onnx,
            output_folder_path=str(f),
            not_use_onnxsim=True,
            verbosity=verbosity,
            output_integer_quantized_tflite=self.args.int8,
            quant_type="per-tensor",  # "per-tensor" (faster) or "per-channel" (slower but more accurate)
            custom_input_op_name_np_data_path=np_data,
        )
        yaml_save(f / "metadata.yaml", self.metadata)  # add metadata.yaml

        # Remove/rename TFLite models
        if self.args.int8:
            tmp_file.unlink(missing_ok=True)
            for file in f.rglob("*_dynamic_range_quant.tflite"):
                file.rename(file.with_name(file.stem.replace("_dynamic_range_quant", "_int8") + file.suffix))
            for file in f.rglob("*_integer_quant_with_int16_act.tflite"):
                file.unlink()  # delete extra fp16 activation TFLite files

        # Add TFLite metadata
        for file in f.rglob("*.tflite"):
            f.unlink() if "quant_with_int16_act.tflite" in str(f) else self._add_tflite_metadata(file)

        return str(f), tf.saved_model.load(f, tags=None, options=None)  # load saved_model as Keras model

    @try_export
    def export_pb(self, keras_model, prefix=colorstr("TensorFlow GraphDef:")):
        """YOLOv8 TensorFlow GraphDef *.pb export https://github.com/leimao/Frozen_Graph_TensorFlow."""
        import tensorflow as tf  # noqa
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2  # noqa

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
        f = self.file.with_suffix(".pb")

        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(m)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)
        return f, None

    @try_export
    def export_tflite(self, keras_model, nms, agnostic_nms, prefix=colorstr("TensorFlow Lite:")):
        """YOLOv8 TensorFlow Lite export."""
        # BUG https://github.com/ultralytics/ultralytics/issues/13436
        import tensorflow as tf  # noqa

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
        saved_model = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        if self.args.int8:
            f = saved_model / f"{self.file.stem}_int8.tflite"  # fp32 in/out
        elif self.args.half:
            f = saved_model / f"{self.file.stem}_float16.tflite"  # fp32 in/out
        else:
            f = saved_model / f"{self.file.stem}_float32.tflite"
        return str(f), None

    @try_export
    def export_edgetpu(self, tflite_model="", prefix=colorstr("Edge TPU:")):
        """YOLOv8 Edge TPU export https://coral.ai/docs/edgetpu/models-intro/."""
        LOGGER.warning(f"{prefix} WARNING ⚠️ Edge TPU known bug https://github.com/ultralytics/ultralytics/issues/1185")

        cmd = "edgetpu_compiler --version"
        help_url = "https://coral.ai/docs/edgetpu/compiler/"
        assert LINUX, f"export only supported on Linux. See {help_url}"
        if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True).returncode != 0:
            LOGGER.info(f"\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}")
            sudo = subprocess.run("sudo --version >/dev/null", shell=True).returncode == 0  # sudo installed on system
            for c in (
                "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -",
                'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | '
                "sudo tee /etc/apt/sources.list.d/coral-edgetpu.list",
                "sudo apt-get update",
                "sudo apt-get install edgetpu-compiler",
            ):
                subprocess.run(c if sudo else c.replace("sudo ", ""), shell=True, check=True)
        ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]

        LOGGER.info(f"\n{prefix} starting export with Edge TPU compiler {ver}...")
        f = str(tflite_model).replace(".tflite", "_edgetpu.tflite")  # Edge TPU model

        cmd = f'edgetpu_compiler -s -d -k 10 --out_dir "{Path(f).parent}" "{tflite_model}"'
        LOGGER.info(f"{prefix} running '{cmd}'")
        subprocess.run(cmd, shell=True)
        self._add_tflite_metadata(f)
        return f, None

    @try_export
    def export_tfjs(self, prefix=colorstr("TensorFlow.js:")):
        """YOLOv8 TensorFlow.js export."""
        check_requirements("tensorflowjs")
        if ARM64:
            # Fix error: `np.object` was a deprecated alias for the builtin `object` when exporting to TF.js on ARM64
            check_requirements("numpy==1.23.5")
        import tensorflow as tf
        import tensorflowjs as tfjs  # noqa

        LOGGER.info(f"\n{prefix} starting export with tensorflowjs {tfjs.__version__}...")
        f = str(self.file).replace(self.file.suffix, "_web_model")  # js dir
        f_pb = str(self.file.with_suffix(".pb"))  # *.pb path

        gd = tf.Graph().as_graph_def()  # TF GraphDef
        with open(f_pb, "rb") as file:
            gd.ParseFromString(file.read())
        outputs = ",".join(gd_outputs(gd))
        LOGGER.info(f"\n{prefix} output node names: {outputs}")

        quantization = "--quantize_float16" if self.args.half else "--quantize_uint8" if self.args.int8 else ""
        with spaces_in_path(f_pb) as fpb_, spaces_in_path(f) as f_:  # exporter can not handle spaces in path
            cmd = (
                "tensorflowjs_converter "
                f'--input_format=tf_frozen_model {quantization} --output_node_names={outputs} "{fpb_}" "{f_}"'
            )
            LOGGER.info(f"{prefix} running '{cmd}'")
            subprocess.run(cmd, shell=True)

        if " " in f:
            LOGGER.warning(f"{prefix} WARNING ⚠️ your model may not work correctly with spaces in path '{f}'.")

        # f_json = Path(f) / 'model.json'  # *.json path
        # with open(f_json, 'w') as j:  # sort JSON Identity_* in ascending order
        #     subst = re.sub(
        #         r'{"outputs": {"Identity.?.?": {"name": "Identity.?.?"}, '
        #         r'"Identity.?.?": {"name": "Identity.?.?"}, '
        #         r'"Identity.?.?": {"name": "Identity.?.?"}, '
        #         r'"Identity.?.?": {"name": "Identity.?.?"}}}',
        #         r'{"outputs": {"Identity": {"name": "Identity"}, '
        #         r'"Identity_1": {"name": "Identity_1"}, '
        #         r'"Identity_2": {"name": "Identity_2"}, '
        #         r'"Identity_3": {"name": "Identity_3"}}}',
        #         f_json.read_text(),
        #     )
        #     j.write(subst)
        yaml_save(Path(f) / "metadata.yaml", self.metadata)  # add metadata.yaml
        return f, None

    def _add_tflite_metadata(self, file):
        """Add metadata to *.tflite models per https://www.tensorflow.org/lite/models/convert/metadata."""
        import flatbuffers

        if ARM64:
            from tflite_support import metadata  # noqa
            from tflite_support import metadata_schema_py_generated as schema  # noqa
        else:
            # TFLite Support bug https://github.com/tensorflow/tflite-support/issues/954#issuecomment-2108570845
            from tensorflow_lite_support.metadata import metadata_schema_py_generated as schema  # noqa
            from tensorflow_lite_support.metadata.python import metadata  # noqa

        # Create model info
        model_meta = schema.ModelMetadataT()
        model_meta.name = self.metadata["description"]
        model_meta.version = self.metadata["version"]
        model_meta.author = self.metadata["author"]
        model_meta.license = self.metadata["license"]

        # Label file
        tmp_file = Path(file).parent / "temp_meta.txt"
        with open(tmp_file, "w") as f:
            f.write(str(self.metadata))

        label_file = schema.AssociatedFileT()
        label_file.name = tmp_file.name
        label_file.type = schema.AssociatedFileType.TENSOR_AXIS_LABELS

        # Create input info
        input_meta = schema.TensorMetadataT()
        input_meta.name = "image"
        input_meta.description = "Input image to be detected."
        input_meta.content = schema.ContentT()
        input_meta.content.contentProperties = schema.ImagePropertiesT()
        input_meta.content.contentProperties.colorSpace = schema.ColorSpaceType.RGB
        input_meta.content.contentPropertiesType = schema.ContentProperties.ImageProperties

        # Create output info
        output1 = schema.TensorMetadataT()
        output1.name = "output"
        output1.description = "Coordinates of detected objects, class labels, and confidence score"
        output1.associatedFiles = [label_file]
        if self.model.task == "segment":
            output2 = schema.TensorMetadataT()
            output2.name = "output"
            output2.description = "Mask protos"
            output2.associatedFiles = [label_file]

        # Create subgraph info
        subgraph = schema.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [input_meta]
        subgraph.outputTensorMetadata = [output1, output2] if self.model.task == "segment" else [output1]
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(model_meta.Pack(b), metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        metadata_buf = b.Output()

        populator = metadata.MetadataPopulator.with_model_file(str(file))
        populator.load_metadata_buffer(metadata_buf)
        populator.load_associated_files([str(tmp_file)])
        populator.populate()
        tmp_file.unlink()

    def _pipeline_coreml(self, model, weights_dir=None, prefix=colorstr("CoreML Pipeline:")):
        """YOLOv8 CoreML pipeline."""
        import coremltools as ct  # noqa

        LOGGER.info(f"{prefix} starting pipeline with coremltools {ct.__version__}...")
        _, _, h, w = list(self.im.shape)  # BCHW

        # Output shapes
        spec = model.get_spec()
        out0, out1 = iter(spec.description.output)
        if MACOS:
            from PIL import Image

            img = Image.new("RGB", (w, h))  # w=192, h=320
            out = model.predict({"image": img})
            out0_shape = out[out0.name].shape  # (3780, 80)
            out1_shape = out[out1.name].shape  # (3780, 4)
        else:  # linux and windows can not run model.predict(), get sizes from PyTorch model output y
            out0_shape = self.output_shape[2], self.output_shape[1] - 4  # (3780, 80)
            out1_shape = self.output_shape[2], 4  # (3780, 4)

        # Checks
        names = self.metadata["names"]
        nx, ny = spec.description.input[0].type.imageType.width, spec.description.input[0].type.imageType.height
        _, nc = out0_shape  # number of anchors, number of classes
        # _, nc = out0.type.multiArrayType.shape
        assert len(names) == nc, f"{len(names)} names found for nc={nc}"  # check

        # Define output shapes (missing)
        out0.type.multiArrayType.shape[:] = out0_shape  # (3780, 80)
        out1.type.multiArrayType.shape[:] = out1_shape  # (3780, 4)
        # spec.neuralNetwork.preprocessing[0].featureName = '0'

        # Flexible input shapes
        # from coremltools.models.neural_network import flexible_shape_utils
        # s = [] # shapes
        # s.append(flexible_shape_utils.NeuralNetworkImageSize(320, 192))
        # s.append(flexible_shape_utils.NeuralNetworkImageSize(640, 384))  # (height, width)
        # flexible_shape_utils.add_enumerated_image_sizes(spec, feature_name='image', sizes=s)
        # r = flexible_shape_utils.NeuralNetworkImageSizeRange()  # shape ranges
        # r.add_height_range((192, 640))
        # r.add_width_range((192, 640))
        # flexible_shape_utils.update_image_size_range(spec, feature_name='image', size_range=r)

        # Print
        # print(spec.description)

        # Model from spec
        model = ct.models.MLModel(spec, weights_dir=weights_dir)

        # 3. Create NMS protobuf
        nms_spec = ct.proto.Model_pb2.Model()
        nms_spec.specificationVersion = 5
        for i in range(2):
            decoder_output = model._spec.description.output[i].SerializeToString()
            nms_spec.description.input.add()
            nms_spec.description.input[i].ParseFromString(decoder_output)
            nms_spec.description.output.add()
            nms_spec.description.output[i].ParseFromString(decoder_output)

        nms_spec.description.output[0].name = "confidence"
        nms_spec.description.output[1].name = "coordinates"

        output_sizes = [nc, 4]
        for i in range(2):
            ma_type = nms_spec.description.output[i].type.multiArrayType
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[0].lowerBound = 0
            ma_type.shapeRange.sizeRanges[0].upperBound = -1
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
            ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
            del ma_type.shape[:]

        nms = nms_spec.nonMaximumSuppression
        nms.confidenceInputFeatureName = out0.name  # 1x507x80
        nms.coordinatesInputFeatureName = out1.name  # 1x507x4
        nms.confidenceOutputFeatureName = "confidence"
        nms.coordinatesOutputFeatureName = "coordinates"
        nms.iouThresholdInputFeatureName = "iouThreshold"
        nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
        nms.iouThreshold = 0.45
        nms.confidenceThreshold = 0.25
        nms.pickTop.perClass = True
        nms.stringClassLabels.vector.extend(names.values())
        nms_model = ct.models.MLModel(nms_spec)

        # 4. Pipeline models together
        pipeline = ct.models.pipeline.Pipeline(
            input_features=[
                ("image", ct.models.datatypes.Array(3, ny, nx)),
                ("iouThreshold", ct.models.datatypes.Double()),
                ("confidenceThreshold", ct.models.datatypes.Double()),
            ],
            output_features=["confidence", "coordinates"],
        )
        pipeline.add_model(model)
        pipeline.add_model(nms_model)

        # Correct datatypes
        pipeline.spec.description.input[0].ParseFromString(model._spec.description.input[0].SerializeToString())
        pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
        pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

        # Update metadata
        pipeline.spec.specificationVersion = 5
        pipeline.spec.description.metadata.userDefined.update(
            {"IoU threshold": str(nms.iouThreshold), "Confidence threshold": str(nms.confidenceThreshold)}
        )

        # Save the model
        model = ct.models.MLModel(pipeline.spec, weights_dir=weights_dir)
        model.input_description["image"] = "Input image"
        model.input_description["iouThreshold"] = f"(optional) IoU threshold override (default: {nms.iouThreshold})"
        model.input_description["confidenceThreshold"] = (
            f"(optional) Confidence threshold override (default: {nms.confidenceThreshold})"
        )
        model.output_description["confidence"] = 'Boxes × Class confidence (see user-defined metadata "classes")'
        model.output_description["coordinates"] = "Boxes × [x, y, width, height] (relative to image size)"
        LOGGER.info(f"{prefix} pipeline success")
        return model

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Execute all callbacks for a given event."""
        for callback in self.callbacks.get(event, []):
            callback(self)


class IOSDetectModel(torch.nn.Module):
    """Wrap an Ultralytics YOLO model for Apple iOS CoreML export."""

    def __init__(self, model, im):
        """Initialize the IOSDetectModel class with a YOLO model and example image."""
        super().__init__()
        _, _, h, w = im.shape  # batch, channel, height, width
        self.model = model
        self.nc = len(model.names)  # number of classes
        if w == h:
            self.normalize = 1.0 / w  # scalar
        else:
            self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])  # broadcast (slower, smaller)

    def forward(self, x):
        """Normalize predictions of object detection model with input size-dependent factors."""
        xywh, cls = self.model(x)[0].transpose(0, 1).split((4, self.nc), 1)
        return cls, xywh * self.normalize  # confidence (3780, 80), coordinates (3780, 4)