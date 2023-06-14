import argparse
import os
import numpy as np
import time

import paddle
import paddle.nn.functional as F
from paddle.inference import Config
from paddle.inference import create_predictor

from PIL import Image, ImageDraw, ImageFont

# import sys
# sys.path.append("../../")
import ppgroundingdino.datasets.transforms as T
from ppgroundingdino.util import box_ops,get_tokenlizer
from ppgroundingdino.util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
)
from ppgroundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
from ppgroundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import ppgroundingdino.util.logger as logger



def print_arguments(args):
    print('-----------  Running Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------')

def load_predictor(model_dir,
                   run_mode='paddle',
                   batch_size=1,
                   device='GPU',
                   cpu_threads=1,
                   enable_mkldnn=False,
                   enable_mkldnn_bfloat16=False,
                   delete_shuffle_pass=False):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16/trt_int8)
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        delete_shuffle_pass (bool): whether to remove shuffle_channel_detect_pass in TensorRT. 
                                    Used by action model.
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need device == 'GPU'.
    """
    if device != 'GPU' and run_mode != 'paddle':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect device=='GPU', but device == {}"
            .format(run_mode, device))
    infer_model = os.path.join(model_dir, 'groundingdino_model.pdmodel')
    infer_params = os.path.join(model_dir, 'groundingdino_model.pdiparams')
  
    config = Config(infer_model, infer_params)
    if device == 'GPU':
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    elif device == 'XPU':
        if config.lite_engine_enabled():
            config.enable_lite_engine()
        config.enable_xpu(10 * 1024 * 1024)
    elif device == 'NPU':
        if config.lite_engine_enabled():
            config.enable_lite_engine()
        config.enable_npu()
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        if enable_mkldnn:
            try:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
                if enable_mkldnn_bfloat16:
                    config.enable_mkldnn_bfloat16()
            except Exception as e:
                print(
                    "The current environment does not support `mkldnn`, so disable mkldnn."
                )
                pass

    # precision_map = {
    #     'trt_int8': Config.Precision.Int8,
    #     'trt_fp32': Config.Precision.Float32,
    #     'trt_fp16': Config.Precision.Half
    # }
    # if run_mode in precision_map.keys():
    #     config.enable_tensorrt_engine(
    #         workspace_size=(1 << 25) * batch_size,
    #         max_batch_size=batch_size,
    #         min_subgraph_size=min_subgraph_size,
    #         precision_mode=precision_map[run_mode],
    #         use_static=False,
    #         use_calib_mode=trt_calib_mode)
    #     if FLAGS.collect_trt_shape_info:
    #         config.collect_shape_range_info(FLAGS.tuned_trt_shape_file)
    #     elif os.path.exists(FLAGS.tuned_trt_shape_file):
    #         print(f'Use dynamic shape file: '
    #               f'{FLAGS.tuned_trt_shape_file} for TRT...')
    #         config.enable_tuned_tensorrt_dynamic_shape(
    #             FLAGS.tuned_trt_shape_file, True)

    #     if use_dynamic_shape:
    #         min_input_shape = {
    #             'image': [batch_size, 3, trt_min_shape, trt_min_shape],
    #             'scale_factor': [batch_size, 2]
    #         }
    #         max_input_shape = {
    #             'image': [batch_size, 3, trt_max_shape, trt_max_shape],
    #             'scale_factor': [batch_size, 2]
    #         }
    #         opt_input_shape = {
    #             'image': [batch_size, 3, trt_opt_shape, trt_opt_shape],
    #             'scale_factor': [batch_size, 2]
    #         }
    #         config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
    #                                           opt_input_shape)
    #         print('trt set dynamic shape done!')

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    if delete_shuffle_pass:
        config.delete_pass("shuffle_channel_detect_pass")
    predictor = create_predictor(config)
    return predictor, config

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * paddle.to_tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box.numpy()
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def load_image(image_path):
  
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image,image_pil

class Predictor(object):
    def __init__(self, args):
        self.tokenizer = get_tokenlizer.get_tokenlizer(args.text_encoder_type)
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
        self.max_text_len = args.max_text_len
        self.box_threshold = args.box_threshold
        self.text_threshold = args.text_threshold
        self.predictor, self.config = load_predictor(args.dino_model_dir)
   
        self.samples = None
        self.caption = None
        self.image_pil = None
        self.tokenized = {}
        self.input_map ={}

           

    def preprocess_image(self,image):
        if isinstance(image, (list, paddle.Tensor)):
            self.samples = nested_tensor_from_tensor_list(image)
       
    
    def preprocess_text(self, text):
        self.tokenized = get_tokenlizer.process_caption(self.tokenizer, text, self.max_text_len)
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            self.tokenized, self.specical_tokens, self.tokenizer
        )
        self.tokenized["position_ids"] = position_ids
        if text_self_attention_masks.shape[1] > self.max_text_len:
                text_self_attention_masks = text_self_attention_masks[
                    :, : self.max_text_len, : self.max_text_len
                ]
                self.tokenized['position_ids'] = position_ids[:, : self.max_text_len]
                self.tokenized["input_ids"] = self.tokenized["input_ids"][:, : self.max_text_len]
                self.tokenized["attention_mask"] = self.tokenized["attention_mask"][:, : self.max_text_len]
        self.tokenized['text_self_attention_masks'] = text_self_attention_masks
      
    def create_inputs(self):

        image,mask = self.samples.decompose()
        self.input_map['image'] = image.numpy()
        self.input_map['mask'] = np.array(mask.numpy(),dtype='int64')

        for key in self.tokenized.keys():
            self.input_map[key] = np.array(self.tokenized[key].numpy(),dtype='int64')
        

    def preprocess(self,arg):
        image,image_pil = load_image(arg.image_file)
        self.image_pil = image_pil
        self.preprocess_image(image[None])

        caption = arg.text_prompt.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        self.caption = caption
        self.preprocess_text(text=[caption])

        self.create_inputs()
    
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            if input_names[i] == 'x':
                input_tensor.copy_from_cpu(self.input_map['image'])
            elif input_names[i] == 'm':
                input_tensor.copy_from_cpu(self.input_map['mask'])
            elif input_names[i] in self.input_map.keys():
                input_tensor.copy_from_cpu(self.input_map[input_names[i]])
        return image_pil
        
    def run(self):
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        pred_boxes =  self.predictor.get_output_handle(output_names[0]).copy_to_cpu()
        pred_logits = self.predictor.get_output_handle(output_names[1]).copy_to_cpu()

        return {"pred_logits":paddle.to_tensor(pred_logits),"pred_boxes":paddle.to_tensor(pred_boxes)}

    def postprocess(self,outputs,with_logits=True):
        
        logits = F.sigmoid(outputs["pred_logits"])[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
      
        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(axis=1) > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenized = self.tokenizer(self.caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, self.tokenizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

def main(args):
    predictor = Predictor(args)
    if args.run_benchmark:
        for i in range(10):
            predictor.preprocess(args)
            result = predictor.run()
            boxes_filt, pred_phrases = predictor.postprocess(result)
    
    if args.run_benchmark:
       predictor.autolog.times.start()  
    predictor.preprocess(args)
    if args.run_benchmark:
       predictor.autolog.times.stamp()

    result = predictor.run()
    if args.run_benchmark:
       predictor.autolog.times.stamp()
    boxes_filt, pred_phrases = predictor.postprocess(result)
    if args.run_benchmark:
       predictor.autolog.times.end(stamp=True)
 
    if args.run_benchmark:
       predictor.autolog.report()
    # make dir
    os.makedirs(args.output_dir, exist_ok=True)

     # visualize pred
    size = predictor.image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    # import ipdb; ipdb.set_trace()
    image_with_box = plot_boxes_to_image(predictor.image_pil, pred_dict)[0]
    image_with_box.save(os.path.join(args.output_dir, "pred.jpg"))


if __name__ == '__main__':
  
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help=("Directory include:'model.pdiparams', 'model.pdmodel', "
              "'infer_cfg.yml', created by tools/export_model.py."),
        required=True)
    parser.add_argument(
        "--image_file", type=str, default=None, help="Path of image file.")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument("--max_text_len", type=int, default=256, help="max text len")
    parser.add_argument("--text_encoder_type", type=str, default="bert-base-uncased", help="text encoder type")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
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
        default="GroundingDINO_SwinT_OGC",
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

