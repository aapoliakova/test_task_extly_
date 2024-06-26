To solve this task I compared 2 methods:
1. [B-LoRA](https://arxiv.org/abs/2403.14572). As this approach is quite effective in case of small dataset and limited computational resources.
2. [DreamBooth](https://arxiv.org/abs/2208.12242) with LoRA. This approach proved yourself as highly effective for context/style transfer, but might require more images for consistent image generation in a new style.   
The results are stored in the folders b-lora images and dreambooth_images respectively. 

In my opinion dreambooth outperforms B-LoRA as this model can produce more colorful images compared to B-LoRA, whereas both models are able to reproduce the main style pattern.  

To run inference, execute the following steps in a new virtual environment:
```
git clone https://github.com/aapoliakova/test_task_extly_.git
cd test_task_extly_
```

```
pip install -r requirements.txt
```



## B-Lora
I finetuned B-LoRA SDXL model for 1000 steps with lr 5e-5 and LoRA rank 64. These are default parameters of HF implementation.
I've also made experiment with lower resolution (512 instead of 1024) and 32 LoRA rank, but the results appeared to be better with the default parameters. 

To run inference you can use LoRA weights loaded on HF or specify the path to checkpoint. 
To replicate the style v34 rare token must be used, please see prompt example below.
Make sure you have a GPU device.
```
python inference.py --prompt="A girl in [v34] style" \
                    --style_LoRA="aapoliakova/v34_style_blora" \
                    --use_blora \
                    --output_path="images" \
                    --num_images_per_prompt=4
```
I also provided code snippet where you can find more configuration details. 
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

## DreamBooth
I finetuned SDXL DreamBooth LoRa for 500 steps with lr 1e-4 and batch size 4. I trained model with prior preservation loss.
To create class images I specified the class prompt as "person portrait" and instance prompt "in v[34] style". 
To run inference run the following code: 
```
python inference.py --prompt="A girl in [v34] style" \
                    --style_LoRA="aapoliakova/v34_style_dreambooth" \
                    --output_path="images" \
                    --num_images_per_prompt=4
```


I also provided code snippets for training with all configuration. 

```
accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"  \
  --dataset_name="dataset_name" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --output_dir="output_dir" \
  --mixed_precision="fp16" \
  --instance_prompt="in v[34] style" \
  --with_prior_preservation \
  --class_data_dir="class_data_dir" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --class_prompt="person portrait" \
  --seed="0"
```

![DreamBooth](https://github.com/aapoliakova/test_task_extly_/blob/master/dreambooth_images/dream_booth.jpg)