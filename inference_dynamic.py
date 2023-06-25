import argparse
import os
import sys

import numpy as np
import paddle
import paddle.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

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
        self.dino_model = GroundingDinoModel.from_pretrained(args.dino_type)
        self.dino_model.eval()

        print(f'sam_model_type {args.sam_model_type}')
        print(f'sam_checkpoint_path {args.sam_checkpoint_path}')
        self.sam_model = SamModel.from_pretrained(args.sam_model_type,input_type=args.sam_input_type)
     
    
    def preprocess(self,image_pil):
        # load image
        image = load_image(image_pil)
        caption = self.args.text_prompt.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        self.caption = caption
        tokenized,position_ids,text_self_attention_masks = preprocess_text(self.dino_model,text=[caption])
        tokenized['position_ids'] = position_ids
        tokenized['text_self_attention_masks'] =text_self_attention_masks
        self.image,self.mask = preprocess_image(image[None])
        self.tokenized = tokenized
        self.image_pil_size = image_pil.size

        image_pil_numpy = np.array(image_pil)
        self.image_seg = self.sam_model.transforms(image_pil_numpy)
        return image_pil

    def get_grounding_output(self,with_logits=True):
        with paddle.no_grad():
            outputs = self.dino_model(self.image,self.mask, input_ids=self.tokenized['input_ids'],
                            attention_mask=self.tokenized['attention_mask'],text_self_attention_masks=self.tokenized['text_self_attention_masks'],
                            position_ids=self.tokenized['position_ids'])
        logits = F.sigmoid(outputs["pred_logits"])[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

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
        seg_masks = self.sam_model(img=self.image_seg,prompt=transformed_boxes)
        
       
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
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
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
