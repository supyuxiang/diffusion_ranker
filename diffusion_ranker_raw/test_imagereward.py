import os
from pathlib import Path
import sys
sys.path.insert(0,'/data/home/yxfeng/ImageReward-main')
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import ImageReward as RM

from utils import load_dit


def rollout_dit():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dit_path = '/data/yxfeng/models/dit_model'
    pipe = load_dit(dit_path,device=device,fp16=True)

    num_samples = 32
    latent_channels = pipe.transformer.config.in_channels
    latent_size = pipe.transformer.config.sample_size
    labels = [281] * num_samples
    gen = torch.Generator(device=device).manual_seed(42)
    
    images = pipe(
        labels,
        num_inference_steps=25,
        generator=gen
    ).images
    return images



if __name__ == "__main__":
    prompt = "a photo of a cat"
    # img_list = ['/data/home/yxfeng/000000222525.jpg','/data/home/yxfeng/image.jpg']
    # model = RM.load("ImageReward-v1.0")
    # with torch.no_grad():
    #     ranking, rewards = model.inference_rank(prompt, img_list)
    #     # Print the result
    #     print("\nPreference predictions:\n")
    #     print(f"ranking = {ranking}")
    #     print(f"rewards = {rewards}")
        
    #     for index in range(len(img_list)):
    #         score = model.score(prompt, img_list[index])
    #         print(f"{img_list[index]:>16s}: {score:.2f}")

    images = rollout_dit()
    save_dir = Path('/data/home/yxfeng/diffusion_ranker/test_images')
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        image.save(save_dir / f'test_image_{i}.png')
    # compute reward
    reward_model = RM.load("ImageReward-v1.0")
    with torch.no_grad():
        for i in range(len(images)):
            score = reward_model.score(prompt, images[i])
            print(f"image {i}: {score:.2f}")
    
























