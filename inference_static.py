import argparse
import os
import numpy as np
import time

import paddle
import paddle.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from deploy.DINOPredictor import Predictor as DINOPredictor
from deploy.SAMPredictor import Predictor as SAMPredictor
import ppgroundingdino.util.logger as logger

def print_arguments(args):
    print('-----------  Running Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------')

def mask_image(image, mask):
    """Mask an image.
    """
    mask_data = np.array(mask, dtype="int32")
    if len(mask_data.shape) == 2: # mode L
        mask_data = np.expand_dims(mask_data, 2)
    masked = np.array(image, dtype="int32") - mask_data
    masked = masked.clip(0, 255).astype("uint8")
    masked = Image.fromarray(masked)
    return masked

def main(args):

    if hasattr(args, 'run_benchmark') and args.run_benchmark:
            import auto_log
            pid = os.getpid()
       
            autolog = auto_log.AutoLogger(
                model_name=args.model_name,
                model_precision='fp32',
                batch_size=1,
                data_shape="dynamic",
                save_path=None,
                inference_config=None,
                pids=pid,
                process_name=None,
                gpu_ids=0,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0,
                logger=logger)

    os.makedirs(args.output_dir, exist_ok=True)

    dino_pipe = DINOPredictor(args)
    sam_pipe = SAMPredictor(args)

    if args.run_benchmark:
        for i in range(50):
            if args.run_benchmark and i>=10:
                autolog.times.start()
            image_pil = dino_pipe.preprocess(args)
            if args.run_benchmark and i>=10:
                autolog.times.stamp()
            result = dino_pipe.run()
            boxes_filt, pred_phrases = dino_pipe.postprocess(result)

            H,W = image_pil.size[1],image_pil.size[0]
            boxes = []
            for box in zip(boxes_filt):
                box = box[0] * paddle.to_tensor([W, H, W, H])
                box[:2] -= box[2:] / 2
                box[2:] += box[:2]
                x0, y0, x1, y1 = box.numpy()
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                boxes.append([x0, y0, x1, y1])
       
            boxes = np.array(boxes)
            mask = sam_pipe.run(np.array(image_pil),
                        {'input_type': args.sam_input_type,
                        'points': None,
                        'boxs': boxes})
            if args.run_benchmark and i>=10:
                autolog.times.stamp()
            pred_mask = Image.fromarray(mask[0][0].astype(np.uint8), mode='P')
            image_masked = mask_image(image_pil, pred_mask)
            if args.run_benchmark and i>=10:
                autolog.times.end(stamp=True)
        if args.run_benchmark:
            autolog.report()

    
    image_pil = dino_pipe.preprocess(args)
    result = dino_pipe.run()
    boxes_filt, pred_phrases = dino_pipe.postprocess(result)

    H,W = image_pil.size[1],image_pil.size[0]
    boxes = []
    for box in zip(boxes_filt):
        box = box[0] * paddle.to_tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        x0, y0, x1, y1 = box.numpy()
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        boxes.append([x0, y0, x1, y1])
       
    boxes = np.array(boxes)
    mask = sam_pipe.run(np.array(image_pil),
                {'input_type': args.sam_input_type,
                'points': None,
                'boxs': boxes})
    
    pred_mask = Image.fromarray(mask[0][0].astype(np.uint8), mode='P')
    image_masked = mask_image(image_pil, pred_mask)
    image_masked.save(os.path.join(args.output_dir, "image_masked.jpg"))


if __name__ == '__main__':
  
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image_file", type=str, default=None, help="Path of image file.")
    parser.add_argument(
        "--dino_model_dir","-dp",
        type=str,
        default=None,
        help=("Directory include:'model.pdiparams', 'model.pdmodel', "
              "'infer_cfg.yml', created by tools/export_model.py."),
        required=True)
    parser.add_argument(
        "--sam_config","-sc",
        dest="cfg",
        help="The config file.",
        default=None,
        type=str,
        required=True)
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "fp16", "int8"],
        help='The tensorrt precision.')
    parser.add_argument(
        "--sam_input_type",
        choices=['boxs', 'points', 'points_grid'],
        required=True,
        help="The model type.",
        type=str)
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument("--max_text_len", type=int, default=256, help="max text len")
    parser.add_argument("--text_encoder_type", type=str, default="bert-base-uncased", help="text encoder type")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_static",
        help="Directory of output visualization files.")
    parser.add_argument(
        "--run_mode",
        type=str,
        default='paddle',
        help="mode of running(paddle/trt_fp32/trt_fp16/trt_int8)")
    parser.add_argument(
        "--device",
        type=str,
        default='GPU',
        help="Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU."
    )
    parser.add_argument(
        "--run_benchmark",
        type=eval,
        default=False,
        help="Whether to log some information about environment, model, configuration and performance."
    )
    parser.add_argument(
        "--model_name",
        default="GroundingDINO_SAM",
        type=str,
        help='When `--benchmark` is True, the specified model name is displayed.'
    )
   

    # paddle.enable_static()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU', 'NPU'
                            ], "device should be CPU, GPU, XPU or NPU"


    main(FLAGS)
