## Prepare

```bash
# install
pip install paddlenlp paddleseg

#Multi-scale deformable attention custom OP compilation
cd ppgroundingdino/models/GroundingDINO/csrc
python setup_ms_deformable_attn_op.py install

```


## dynamic inference
```bash

python inference_dynamic.py -dt groundingdino-swint-ogc  --sam_model_type SamVitH  -sp sam_vit_h.pdparams --sam_input_type boxs -i 498604_0_final.png -o output -t "eye"
```

## Export model for static inference
```bash
#export grounding dino model
python export_groundingdino.py -c groundingdino-swint-ogc

#export sam model
python export_sam.py --model_type SamVitH --input_type boxs --save_dir export_output -p sam_vit_h.pdparams 

#inference
python inference_static.py  --image_file 498604_0_final.png -t "eye" -dp output_groundingdino -sc export_output_SamVitH_boxs/deploy.yaml --sam_input_type boxs

```

## Benchmark

```bash

python inference_dynamic.py -dt groundingdino-swint-ogc  --sam_model_type SamVitH  -sp sam_vit_h.pdparams --sam_input_type boxs -i 498604_0_final.png -o output -t "eye" --run_benchmark=True

#static inference
python inference_static.py  --image_file 498604_0_final.png -t "eye" -dp output_groundingdino -sc export_output_SamVitH_boxs/deploy.yaml --sam_input_type boxs --run_benchmark=True

```