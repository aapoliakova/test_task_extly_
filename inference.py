import argparse

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

from utils_blora import BLOCKS, filter_lora, scale_lora
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", type=str, required=True, help="prompt in a format: a {context} in v[34] style"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="path to save the images"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=4, help="number of images per prompt"
    )
    parser.add_argument(
        "--style_LoRA", type=str, default=None, help="path for the style B-LoRA or Dreambooth"
    )
    parser.add_argument(
        "--use_blora",
        default=False,
        action="store_true",
        help="Flag to use B-LoRA instead of dreambooth",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                         vae=vae,
                                                         torch_dtype=torch.float16).to("cuda")

    if args.use_blora:
        style_B_LoRA_sd, _ = pipeline.lora_state_dict(args.style_LoRA)
        style_B_LoRA = filter_lora(style_B_LoRA_sd, BLOCKS['style'])
        style_B_LoRA = scale_lora(style_B_LoRA, 1.0)
        pipeline.load_lora_into_unet(style_B_LoRA, None, pipeline.unet)
    else:
        pipeline.load_lora_weights(args.style_LoRA,
                                   weight_name="pytorch_lora_weights.safetensors")

    images = pipeline(args.prompt, num_images_per_prompt=args.num_images_per_prompt).images
    Path(args.output_path).mkdir(exist_ok=True, parents=True)
    for i, img in enumerate(images):
        img.save(f'{args.output_path}/{args.prompt}_{i}.jpg')
