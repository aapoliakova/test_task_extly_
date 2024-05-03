To solve this task I have chosen [B-LoRA](https://arxiv.org/abs/2403.14572) method as this approach works well for 
style transfer in case of small dataset and limited computational resources. 
This method is based on the observation that only several Attention blocks of SDXL need to be fine-tuned with LoRa 
adapters to preserve style of the image or an instance.

The training script with all arguments: 

```
 accelerate launch train_dreambooth_b-lora_sdxl.py \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
 --dataset_name="aapoliakova/style_v143" \
 --output_dir="output_dir_blora" \
 --instance_prompt="v[43]" \
 --resolution=512 \
 --rank=32 \
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

As you can see I trained model for 1000 steps with learning rate 5e-5. An style token is v43. 
I have chosen image resolution 512 and LoRa rank 32 to fit model into memory. 

