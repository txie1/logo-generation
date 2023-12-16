import os
import argparse

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from datasets import load_dataset
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from torchinfo import summary

from scheduler import DDIMScheduler
from model import UNet
from utils import save_images, normalize_to_neg_one_to_one, plot_losses
from dataset import CustomDataset
import pandas as pd

n_timesteps = 1000
n_inference_timesteps = 50


def main(args):
    model = UNet(3,
                 image_size=args.resolution,
                 hidden_dims=[128, 256, 512, 1024],
                 use_linear_attn=False)
    noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps,
                                    beta_schedule="cosine")

    if args.pretrained_model_path:
        pretrained = torch.load(args.pretrained_model_path)["model_state"]
        model.load_state_dict(pretrained, strict=False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    augmentations = Compose([
        Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(args.resolution),
        RandomHorizontalFlip(),
        ToTensor(),
    ])

    if args.dataset_name is not None:

        def transforms(examples):
            images = [
                augmentations(image.convert("RGB"))
                for image in examples["image"]
            ]
            return {"input": images}

        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
        dataset.set_transform(transforms)
    else:
        df = pd.read_pickle(args.train_data_path)
        dataset = CustomDataset(df, augmentations)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True)

    device = torch.device(f"cuda:{args.gpu}")
    summary(model, [(1, 3, args.resolution, args.resolution), (1, )], verbose=1)
    model = model.to(device)

    loss_fn = F.l1_loss if args.use_l1_loss else F.mse_loss
    global_step = 0
    losses = []
    for epoch in range(951, args.num_epochs+1):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        losses_log = 0
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"].to(device)
            clean_images = normalize_to_neg_one_to_one(clean_images)

            batch_size = clean_images.shape[0]
            noise = torch.randn(clean_images.shape).to(device)
            timesteps = torch.randint(0,
                                      noise_scheduler.num_train_timesteps,
                                      (batch_size, ),
                                      device=device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise,
                                                     timesteps)

            noise_pred = model(noisy_images, timesteps)["sample"]
            loss = loss_fn(noise_pred, noise)
            loss.backward()

            if args.use_clip_grad:
                clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            losses_log += loss.detach().item()
            logs = {
                "loss_avg": losses_log / (step + 1),
                "loss": loss.detach().item(),
                "lr": args.learning_rate,
                "step": global_step
            }

            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()
        losses.append(losses_log / (step + 1))


        # Generate sample images for visual inspection
        if epoch % 50 == 0:
            with torch.no_grad():
                generated_images = noise_scheduler.generate(
                    model,
                    num_inference_steps=n_inference_timesteps,
                    generator=None,
                    eta=1.0,
                    use_clipped_model_output=True,
                    batch_size=args.eval_batch_size,
                    output_type="numpy",
                    device=device)

                save_images(generated_images, epoch, args)
                plot_losses(losses, f"{args.loss_logs_dir}/{epoch}/")

        #-------------------------------#
        # Saves checkpoints       
        if epoch % args.save_model_epochs == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_path = f"{args.output_dir}/ckpt_{epoch}_{step}.pth"
            torch.save({'model_state': model.state_dict(),}, ckpt_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default="logo-wizard/modern-logo-dataset")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_data_path",
                        type=str,
                        default=None,
                        help="A df containing paths to training images.")
    parser.add_argument("--output_dir",
                        type=str,
                        default="trained_models/logo_test")
    parser.add_argument("--samples_dir", type=str, default="test_samples/logo")
    parser.add_argument("--loss_logs_dir", type=str, default="training_logs/logo")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=32) #16
    parser.add_argument("--eval_batch_size", type=int, default=32) #32
    parser.add_argument("--num_epochs", type=int, default=1000) #100
    parser.add_argument("--save_model_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_clip_grad", type=bool, default=False)
    parser.add_argument("--use_l1_loss", type=bool, default=False)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--pretrained_model_path",
                        type=str,
                        default=None,
                        help="Path to pretrained model")


    args = parser.parse_args()

    if args.dataset_name is None and args.train_data_path is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory."
        )

    main(args)