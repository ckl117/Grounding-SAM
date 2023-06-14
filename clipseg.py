import os
import sys

rootdir = os.path.abspath(os.path.dirname(__file__))
#print(rootdir)

#sys.path.append(rootdir + "/../../")

from paddlenlp.transformers import CLIPSegProcessor
from paddlenlp.transformers import CLIPSegTextConfig, CLIPSegConfig
from paddlenlp.transformers import CLIPSegForImageSegmentation
import paddle
import paddle.nn.functional as F
from PIL import Image, ImageFilter
import numpy as np
import yake
import time

class ClipSegInfer():
    def __init__(self, model_name_or_path="CIDAS/clipseg-rd64-refined"):
        self.model_name_or_path = model_name_or_path
        self.config = CLIPSegConfig.from_pretrained(self.model_name_or_path)
        self.processor = CLIPSegProcessor.from_pretrained(self.model_name_or_path)
        self.model = CLIPSegForImageSegmentation.from_pretrained(self.model_name_or_path)
        self.model.eval()
        language = "en"
        max_ngram_size = 3
        deduplication_threshold = 0.9
        numOfKeywords = 2
        self.custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)

    def run(self, image, texts):
        #t1 = time.time()
        inputs = self.processor(text=[texts[0]]*2, images=[image]*2, padding=True, return_tensors="pd")
        with paddle.no_grad():
            outputs = self.model(**inputs, return_dict=True)
        logits = outputs.logits
        logits = logits.unsqueeze(1)
        mask = logits[0].numpy().astype("uint8")
        _,h,w = mask.shape
        mask = mask.reshape((h, w))
        mask = Image.fromarray(mask).convert("L")
        #t2 = time.time()
        #print(f"clipseg cost {t2 -t1}")
        return mask


def get_masked_img(img, mask):
    img_np = np.array(img)
    mask_np = np.array(mask)
    mask_np = mask_np / 255.0
    mask_np[mask_np > 0.5] = 1
    mask_np[mask_np <= 0.5] = 0
    mask_np = np.expand_dims(mask_np, -1)
    #print(img_np.shape)
    #print(mask_np.shape)
    masked_img = img_np * mask_np
    print(masked_img.shape)
    masked_img = Image.fromarray(masked_img.astype('uint8'))
    return masked_img


if __name__ == '__main__':
    model_name_or_path = "./CIDAS/clipseg-rd64-refined"
    clipseginfer = ClipSegInfer(model_name_or_path)
    
    img = Image.open("498604_0_final.png").convert('RGB')
    text = "eye"
    texts = [text]
    img_w, img_h = img.size
    total = 0.0
    for i in range(50):
        if i>=10:
            st = time.time()
        mask = clipseginfer.run(img, texts)
        if i>=10:
            total += time.time()-st
        mask = mask.resize((img_w, img_h))
        masked_img = get_masked_img(img, mask)

        mask.save('clipseg_mask.png')
        masked_img.save('clipseg_masked.png')
        
    print("cost:",total/40)