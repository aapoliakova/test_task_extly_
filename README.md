To solve this task I have chosen [B-LoRA](https://arxiv.org/abs/2403.14572) method as this approach works well for 
style transfer in case of small dataset and limited computational resources. 
This method is based on the observation that only several Attention blocks of SDXL need to be fine-tuned with LoRa 
adapters to preserve style of the image or an instance.

The training script with all arguments: 

```
accelerate launch train_dreambooth_b-lora_sdxl.py \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
 --dataset_name="dataset_name" \
 --output_dir="output_dir" \
 --instance_prompt="v[43]" \
 --resolution=1024 \
 --rank=64 \
 --train_batch_size=1 \
 --learning_rate=5e-5 \
 --lr_scheduler="constant" \
 --lr_warmup_steps=0 \
 --max_train_steps=1000 \
 --checkpointing_steps=500 \
 --seed="0" \
 --gradient_checkpointing \
 --mixed_precision="fp16"
```
I fine-tuned the model with default parameters as recommended by the authors, I encoded the needed style with rare token v43. 

To run inference, execute the following steps in a new virtual environment:
```
https://github.com/aapoliakova/test_task_extly_.git
cd test_task_extly_
```

```
pip install -r requirements.txt
```

```
python inference.py --prompt="promt" \
    --style_B_LoRA="aapoliakova/exctly_test_style" \
    --output_path="images" \
    --num_images_per_prompt=4
```
