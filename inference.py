import argparse

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

from utils_blora import BLOCKS, filter_lora, scale_lora
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", type=str, required=True, help="B-LoRA prompt"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="path to save the images"
    )
    parser.add_argument(
        "--style_B_LoRA", type=str, default=None, help="path for the style B-LoRA"
    )
    parser.add_argument(
        "--style_alpha", type=float, default=1., help="alpha parameter to scale the style B-LoRA weights"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=4, help="number of images per prompt"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                         vae=vae,
                                                         torch_dtype=torch.float16).to("cuda")

    if args.style_B_LoRA is not None:
        style_B_LoRA_sd, _ = pipeline.lora_state_dict(args.style_B_LoRA)
        style_B_LoRA = filter_lora(style_B_LoRA_sd, BLOCKS['style'])
        style_B_LoRA = scale_lora(style_B_LoRA, args.style_alpha)
    else:
        style_B_LoRA = {}

    pipeline.load_lora_into_unet(style_B_LoRA, None, pipeline.unet)
    images = pipeline(args.prompt, num_images_per_prompt=args.num_images_per_prompt).images

    Path(args.output_path).mkdir(exist_ok=True, parents=True)

    for i, img in enumerate(images):
        img.save(f'{args.output_path}/{args.prompt}_{i}.jpg')
