import argparse
import os
import sys

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.inference import Config
from paddle.inference import create_predictor

from PIL import Image, ImageDraw, ImageFont
import time
import ppgroundingdino.datasets.transforms as T
from ppgroundingdino.models.GroundingDINO.groundingdino import GroundingDinoModel
from ppgroundingdino.util import box_ops,get_tokenlizer
from ppgroundingdino.util.slconfig import SLConfig
from ppgroundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from ppgroundingdino.util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
)
from ppgroundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
import ppgroundingdino.util.logger as logger

# from segment_anything.predictor import SamPredictor
# from segment_anything.build_sam import sam_model_registry

from segment_anything.modeling.sam_models import SamModel
from paddle.utils.cpp_extension import load
# jit compile custom op
# ms_deformable_attn = load(
#     name="deformable_detr_ops",
#     sources=["./ppgroundingdino/models/GroundingDINO/csrc/ms_deformable_attn_op.cc",
#     "./ppgroundingdino/models/GroundingDINO/csrc/ms_deformable_attn_op.cu"])

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"函数 {func.__name__} 运行时间：{execution_time} 秒")
        return result
    return wrapper

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

def load_image(image_pil):
    # load image
    #image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image

class DINO(object):
    def __init__(self):
        self.tokenizer = get_tokenlizer.get_tokenlizer("bert-base-uncased")
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
        args.max_text_len = 256
        args.box_threshold = 0.3
        args.text_threshold = 0.25
        self.max_text_len = args.max_text_len
        self.box_threshold = args.box_threshold
        self.text_threshold = args.text_threshold
        self.samples = None
        self.caption = None
        self.image_pil = None
        self.tokenized = {}
        self.input_map ={}

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
                position_ids = position_ids[:, : self.max_text_len]
                self.tokenized["input_ids"] = self.tokenized["input_ids"][:, : self.max_text_len]
                self.tokenized["attention_mask"] = self.tokenized["attention_mask"][:, : self.max_text_len]
        # self.tokenized['text_self_attention_masks'] = text_self_attention_masks
        return self.tokenized,position_ids,text_self_attention_masks
        

    def preprocess(self,arg, image_pil):
        
        image = load_image(image_pil)
        self.image_pil = image_pil
        self.preprocess_image(image[None])

        caption = arg.text_prompt.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        self.caption = caption
        self.preprocess_text(text=[caption])

        self.create_inputs()

        


def preprocess_text(model,text):
    max_text_len = model.max_text_len
    tokenized = get_tokenlizer.process_caption(model.tokenizer, text, max_text_len)
   
    (
        text_self_attention_masks,
        position_ids,
        cate_to_token_mask_list,
    ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, model.specical_tokens, model.tokenizer
        )
    if text_self_attention_masks.shape[1] > model.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : max_text_len, : max_text_len
            ]
            position_ids = position_ids[:, : max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : max_text_len]
            #tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : max_text_len]
            
    return tokenized,position_ids,text_self_attention_masks

def preprocess_image(image):
    if isinstance(image, (list, paddle.Tensor)):
        samples = nested_tensor_from_tensor_list(image)
    return samples.decompose()


def load_dino(model_dir,
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
        config.enable_use_gpu(4000, 0)
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

    # disable print log when predict
    #config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    
    shape_range_file = infer_model + "shape.txt"
    # config.collect_shape_range_info(shape_range_file)

    # config.switch_ir_debug()

    config.enable_tuned_tensorrt_dynamic_shape(shape_range_file, True)
    # config.enable_tensorrt_engine(
    #     workspace_size=1 << 30,
    #     precision_mode=paddle.inference.PrecisionType.Half,
    #     max_batch_size=0,
    #     min_subgraph_size=30,
    #     use_static=True,
    #     use_calib_mode=False)
    config.delete_pass("add_support_int8_pass")
    config.delete_pass("trt_skip_layernorm_fuse_pass")
    config.delete_pass("preln_residual_bias_fuse_pass")
    config.exp_disable_tensorrt_ops(["pad3d", "set_value", "reduce_all"])
    # disable feed, fetch OP, needed by zero_copy_run
    pass_builder = config.pass_builder()
    # pass_builder.set_passes(["inplace_op_var_pass"])
    config.switch_use_feed_fetch_ops(False)
    if delete_shuffle_pass:
        config.delete_pass("shuffle_channel_detect_pass")
    predictor = create_predictor(config)
    return predictor

def load_sam(model_dir):
    model_path = './export_output_SamVitH_boxs/model.pdmodel'
    param_path = './export_output_SamVitH_boxs/model.pdiparams'
    model_path = os.path.join(model_dir, 'model.pdmodel')
    param_path = os.path.join(model_dir, 'model.pdiparams')
    pred_cfg = Config(model_path, param_path)
    pred_cfg.enable_memory_optim(True)
    pred_cfg.switch_ir_optim(True)
    pred_cfg.enable_use_gpu(4000, 0)

    shape_range_file = model_path + "shape.txt"
    # pred_cfg.collect_shape_range_info(shape_range_file)
    # pred_cfg.switch_ir_debug()

    # pred_cfg.enable_tuned_tensorrt_dynamic_shape(shape_range_file, True)
    # pred_cfg.enable_tensorrt_engine(
    #     workspace_size=1 << 35,
    #     precision_mode=paddle.inference.PrecisionType.Half,
    #     max_batch_size=0,
    #     min_subgraph_size=30,
    #     use_static=True,
    #     use_calib_mode=False)
    pred_cfg.exp_disable_tensorrt_ops(["concat_1.tmp_0", "set_value"])
    pred_cfg.delete_pass("add_support_int8_pass")
    pred_cfg.delete_pass("shuffle_channel_detect_pass")
    pred_cfg.delete_pass("trt_skip_layernorm_fuse_pass")
    pred_cfg.delete_pass("preln_residual_bias_fuse_pass")
    pass_builder = pred_cfg.pass_builder()
    # pass_builder.set_passes(["inplace_op_var_pass"])
    predictor = create_predictor(pred_cfg)
    return predictor

class DinoSamInfer():
    def __init__(self,args):
        # cfg
        self.args = args
        self.caption = None
        self.image_pil_size = None

        self.text_threshold = args.text_threshold
        self.box_threshold = args.box_threshold
        # load model
        print(f'dino_model {args.dino_type}')
        self.dino_model = DINO()
        print(f'sam_model_type {args.sam_model_type}')
        print(f'sam_checkpoint_path {args.sam_checkpoint_path}')
        self.sam_model = SamModel.from_pretrained(args.sam_model_type,input_type=args.sam_input_type)

        self.predictor_dino = load_dino('./output_groundingdino')

        self.predictor_sam = load_sam('./export_output_SamVitH_boxs')
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

    def preprocess(self,image_pil):
   
        # load image
        image = load_image(image_pil)
        caption = self.args.text_prompt.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        self.caption = caption
        # tokenized,position_ids,text_self_attention_masks = preprocess_text(self.dino_model,text=[caption])
        tokenized,position_ids,text_self_attention_masks = self.dino_model.preprocess_text([caption])
        tokenized['position_ids'] = position_ids
        tokenized['text_self_attention_masks'] =text_self_attention_masks
        self.image,self.mask = preprocess_image(image[None])
        self.tokenized = tokenized
        self.image_pil_size = image_pil.size

        image_pil_numpy = np.array(image_pil)
        self.image_seg = self.sam_model.transforms(image_pil_numpy)
        return image_pil


    def get_grounding_output(self,with_logits=True):
        paddle.set_flags({'FLAGS_set_to_1d' : False})
        self.mask = paddle.cast(self.mask, dtype='int64')
        self.tokenized["text_self_attention_masks"] = paddle.cast(self.tokenized["text_self_attention_masks"], dtype='int64')


        
        [pred_boxes, pred_logits] = self.predictor_dino.run([self.image,self.mask, self.tokenized['input_ids'],self.tokenized['attention_mask'],self.tokenized['text_self_attention_masks'],self.tokenized['position_ids']])
        
        # 推理
        for i in range(10):
            [pred_boxes, pred_logits] = self.predictor_dino.run([self.image,self.mask, self.tokenized['input_ids'],self.tokenized['attention_mask'],self.tokenized['text_self_attention_masks'],self.tokenized['position_ids']])
            paddle.device.cuda.synchronize()
        total_time = 0
        for i in range(40):
            start = time.time()
            [pred_boxes, pred_logits] = self.predictor_dino.run([self.image,self.mask, self.tokenized['input_ids'],self.tokenized['attention_mask'],self.tokenized['text_self_attention_masks'],self.tokenized['position_ids']])
            paddle.device.cuda.synchronize()
            end = time.time()
            time.sleep(0.1)
            total_time = total_time + end - start
        print(f'dino infer time = {total_time * 1000 / 40} ms')

        logits = F.sigmoid(pred_logits)[0]  # (nq, 256)
        boxes = pred_boxes[0]  # (nq, 4)

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(axis=1) > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = self.dino_model.tokenizer
        tokenized = tokenlizer(self.caption)
        # build pred

        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit >self.text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

    def run(self):
        
        # run dino model
        boxes_filt, pred_phrases = self.get_grounding_output()
        H,W = self.image_pil_size[1],self.image_pil_size[0]
        boxes = []
        for box in zip(boxes_filt):
            box = box[0] * paddle.to_tensor([W, H, W, H])
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box.numpy()
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            boxes.append([x0, y0, x1, y1])
       
        boxes = np.array(boxes)

        transformed_boxes = self.sam_model.preprocess_prompt(point_coords=None, point_labels=None, box=boxes)
        paddle.set_flags({'FLAGS_set_to_1d' : True})
        transformed_boxes = paddle.squeeze(transformed_boxes, axis=0)

        seg_masks = self.predictor_sam.run([self.image_seg, transformed_boxes])
        
        # 推理
        for i in range(10):
            seg_masks = self.predictor_sam.run([self.image_seg, transformed_boxes])
            paddle.device.cuda.synchronize()
        total_time = 0
        for i in range(40):
            start = time.time()
            seg_masks = self.predictor_sam.run([self.image_seg, transformed_boxes])
            paddle.device.cuda.synchronize()
            end = time.time()
            total_time = total_time + end - start
            time.sleep(0.2)
        print(f'sam infer time = {total_time * 1000 / 40} ms')

        seg_masks = seg_masks[0]
               
        return seg_masks

    def postprocess(self,mask):
        mask = self.sam_model.postprocess(mask)
        masks = np.array(mask[:,0,:,:])
        init_mask = np.zeros(masks.shape[-2:])
        for mask in masks:
            mask = mask.reshape(mask.shape[-2:])
            mask[mask == False] = 0
            mask[mask == True] = 1
            init_mask += mask

        init_mask[init_mask == 0] = 0
        init_mask[init_mask != 0] = 255
        #init_mask = 255 - init_mask
        init_mask = Image.fromarray(init_mask).convert('L')
    
        return init_mask


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--dino_type", "-dt", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--sam_model_type",
        #choices=['SamVitL', 'SamVitB', 'SamVitH'],
        required=True,
        help="The model type.",
        type=str)
    parser.add_argument(
        "--sam_checkpoint_path", "-sp", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_input_type",
        choices=['boxs', 'points', 'points_grid'],
        required=True,
        help="The model type.",
        type=str)
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="my_static", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    parser.add_argument(
        "--run_benchmark",
        type=eval,
        default=False,
        help="Whether to log some information about environment, model, configuration and performance."
    )

    args = parser.parse_args()

    pipe = DinoSamInfer(args)

    if args.run_benchmark:
        import auto_log
        pid = os.getpid()

        autolog = auto_log.AutoLogger(
            model_name='goundingdino-sam',
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

    # make dir
    os.makedirs(args.output_dir, exist_ok=True)
    image_pil = Image.open(args.image_path).convert("RGB")
    if args.run_benchmark:
        for i in range(50):
            if args.run_benchmark and i>=10:
                autolog.times.start()
            pipe.preprocess(image_pil)
            if args.run_benchmark and i>=10:
                autolog.times.stamp()
            seg_masks = pipe.run()
            #seg_masks = paddle.rand([2,3,256,256])
            if args.run_benchmark and i>=10:
                autolog.times.stamp()
            init_mask = pipe.postprocess(seg_masks)
            if args.run_benchmark and i>=10:
                autolog.times.end(stamp=True)
        if args.run_benchmark:
            autolog.report()
  

    pipe.preprocess(image_pil)
    seg_masks = pipe.run()
    init_mask = pipe.postprocess(seg_masks)
    
    image_masked = mask_image(image_pil, init_mask)
    image_masked.save(os.path.join(args.output_dir, "image_masked.jpg"))
