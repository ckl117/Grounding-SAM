import os
import numpy as np
import paddle
import paddle.nn.functional as F
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import yake
import time
import random
import ppgroundingdino.datasets.transforms as T
from ppgroundingdino.models import build_model
from ppgroundingdino.util.slconfig import SLConfig
from ppgroundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from segment_anything.predictor import SamPredictor
from segment_anything.build_sam import sam_model_registry

curdir = os.path.abspath(os.path.dirname(__file__))

def load_model(model_config_path, model_checkpoint_path):
    args = SLConfig.fromfile(model_config_path)
    model = build_model(args)
    checkpoint = paddle.load(model_checkpoint_path, return_numpy=True)
    load_res = model.set_state_dict(clean_state_dict(checkpoint))
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    with paddle.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = F.sigmoid(outputs["pred_logits"])[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(axis=1) > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

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

organ_str = """"""


class DinoSamInfer():
    def __init__(self):
        # cfg
        config_file = curdir + "/ppgroundingdino/config/GroundingDINO_SwinT_OGC.py"
        checkpoint_path = curdir + "/groundingdino_swint_ogc.pdparams"
    
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333), 
            T.ToTensor(), 
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # load model
        print(f'dino_model {config_file}')
        print(f'checkpoint_path {checkpoint_path}')
        self.dino_model = load_model(config_file, checkpoint_path)
        self.sam_model = sam_model_registry['vit_h'](checkpoint=curdir + '/sam_vit_h.pdparams')
        self.sam_predictor = SamPredictor(self.sam_model)

    def run(self, image_pil, text_prompt):
        t1 = time.time()

        image, _ = self.transform(image_pil, None)

        # run dino model
        boxes_filt, pred_phrases = get_grounding_output(self.dino_model, image, text_prompt, 0.3, 0.25)
        size = image_pil.size

        H, W = size[1], size[0]
        boxes = []
        for box in zip(boxes_filt):
            box = box[0] * paddle.to_tensor([W, H, W, H])
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box.numpy()
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            boxes.append([x0, y0, x1, y1])

        boxes = np.array(boxes)

        image_pil_numpy = np.array(image_pil)
        self.sam_predictor.set_image(image_pil_numpy)

        transformed_boxes = paddle.to_tensor(self.sam_predictor.transform.apply_boxes(boxes, size))

        masks, _, _ = self.sam_predictor.predict_paddle(
            point_coords=None, 
            point_labels=None, 
            boxes=transformed_boxes,
            multimask_output=False)

        masks = np.array(masks)
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

        t2 = time.time()
        print(f"dino cost time {t2 -t1}")

        image_masked = mask_image(image_pil, init_mask)
        return init_mask, image_masked


if __name__ == "__main__":
    pipe = DinoSamInfer()
    for i in range(5):
        img = Image.open("498604_0_final.png")
        text = "eye"
        mask, masked = pipe.run(img, text)
        mask.save('mask.png')
        masked.save('masked.png')