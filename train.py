import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from types import SimpleNamespace
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from model import MultiStage_denoise
from multi_resolution_stft_loss import MultiResolutionSTFTLoss
from mr_spectrogram_dataset_loader import MRSpectrogramDatasetLoader


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_unet_args():
    return SimpleNamespace(
        depth=6,
        num_tfc=3,
        use_SAM=True,
        use_fencoding=True,
        f_dim=513,
        num_stages=2
    )


def freeze_encoder_blocks(model: MultiStage_denoise, freeze_until_idx: int):
    if freeze_until_idx == 0:
        return []

    print(f'Freezing encoder blocks [0..{freeze_until_idx - 1}]')
    frozen_blocks = []

    for i in range(freeze_until_idx):
        block_s1 = model.encoder_s1.eblocks[i]
        for param in block_s1.parameters():
            param.requires_grad = False
        frozen_blocks.append(block_s1)

        if hasattr(model, 'encoder_s2'):
            block_s2 = model.encoder_s2.eblocks[i]
            for param in block_s2.parameters():
                param.requires_grad = False
            frozen_blocks.append(block_s2)

    return frozen_blocks


def calculate_loss(output,
                   clean,
                   l1_loss,
                   stft_loss,
                   use_stage1_loss):
    if isinstance(output, tuple):
        output_stage2, output_stage1 = output

        loss = l1_loss(output_stage2, clean) + stft_loss(output_stage2, clean)

        if use_stage1_loss:
            loss += 0.5 * l1_loss(output_stage1, clean)
    else:
        loss = l1_loss(output, clean) + stft_loss(output, clean)

    return loss


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def train(model: MultiStage_denoise | DistributedDataParallel,
          train_loader: DataLoader,
          val_loader: DataLoader,
          epochs: int,
          lr: float,
          weight_decay: float,
          frozen_patience: int,
          patience_after_all_unfrozen: int,
          device: torch.device,
          saved_checkpoints_folder: str,
          saved_metrics_json_path: str,
          fine_tune: bool,
          use_stage1_loss: bool = True,
          saved_checkpoint_counter=0):
    model_ = unwrap_model(model)
    frozen_blocks = freeze_encoder_blocks(model_, freeze_until_idx=3 if fine_tune else 0)

    scaler = GradScaler() if device.type == 'cuda' else None

    optimizer = optim.Adam(
        [{'params': [p for p in model.parameters() if p.requires_grad]}],
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )

    l1_loss = nn.L1Loss()
    stft_loss = MultiResolutionSTFTLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    final_patience_counter = 0
    current_unfreeze_index = 0

    start_epoch = 0
    metrics_history = []

    if os.path.exists(saved_metrics_json_path):
        with open(saved_metrics_json_path, 'r') as f:
            metrics_history = json.load(f)

        last_metrics = metrics_history[-1]

        start_epoch = last_metrics['epoch']
        for g in optimizer.param_groups:
            g['lr'] = last_metrics['learning_rate']
        best_val_loss = last_metrics['best_val_loss_so_far']
        current_unfreeze_index = last_metrics['unfrozen_blocks']

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0

        for noisy, clean in tqdm(train_loader, desc=f'[Epoch {epoch+1}/{epochs}]'):
            noisy = torch.tensor(noisy, dtype=torch.float32).to(device)
            clean = torch.tensor(clean, dtype=torch.float32).to(device)

            with autocast(device_type=device.type, enabled=(scaler is not None)):
                output = model(noisy)
                loss = calculate_loss(output, clean, l1_loss, stft_loss, use_stage1_loss)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f'Train Loss: {avg_train_loss:.4f}')

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for noisy, clean in tqdm(val_loader):
                noisy = torch.tensor(noisy, dtype=torch.float32).to(device)
                clean = torch.tensor(clean, dtype=torch.float32).to(device)

                with autocast(device_type=device.type, enabled=(scaler is not None)):
                    output = model(noisy)
                    loss = calculate_loss(output, clean, l1_loss, stft_loss, use_stage1_loss)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_filepath = os.path.join(saved_checkpoints_folder, f'checkpoint_{saved_checkpoint_counter}.pth')
            torch.save(model.state_dict(), checkpoint_filepath)
            print(f'Saved best model → {checkpoint_filepath}')

            patience_counter = 0
            final_patience_counter = 0
            saved_checkpoint_counter += 1
        else:
            if current_unfreeze_index < len(frozen_blocks):
                patience_counter += 1
                print(f'Patience: {patience_counter}/{frozen_patience}')
                if patience_counter >= frozen_patience:
                    block = frozen_blocks[current_unfreeze_index]
                    for param in block.parameters():
                        param.requires_grad = True
                    optimizer.add_param_group({'params': block.parameters()})
                    print(f'Unfroze block {current_unfreeze_index} at epoch {epoch + 1}')
                    current_unfreeze_index += 1
                    patience_counter = 0
            else:
                final_patience_counter += 1
                print(f'Final patience: {final_patience_counter}/{patience_after_all_unfrozen}')
                if final_patience_counter >= patience_after_all_unfrozen:
                    print(f'Early stopping: no improvement for {patience_after_all_unfrozen} epochs')
                    break

        scheduler.step(avg_val_loss)

        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': optimizer.param_groups[0]["lr"],
            'best_val_loss_so_far': best_val_loss,
            'unfrozen_blocks': current_unfreeze_index
        })

        with open(saved_metrics_json_path, 'w') as f:
            json.dump(metrics_history, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MultiStage Denoising U-Net")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--frozen_patience", type=int, default=5)
    parser.add_argument("--patience_after_all_unfrozen", type=int, default=5)

    parser.add_argument("--segment_frames", type=float, default=258)
    parser.add_argument("--hop_frames", type=float, default=258)

    parser.add_argument("--use_stage1_loss", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--fine_tune", action="store_true")

    parser.add_argument("--train_dataset_clean_dir", type=str)
    parser.add_argument("--train_dataset_noisy_dir", type=str)
    parser.add_argument("--val_dataset_clean_dir", type=str)
    parser.add_argument("--val_dataset_noisy_dir", type=str)

    parser.add_argument("--pretrained_path", type=str, default="checkpoint_denoiser")
    parser.add_argument("--saved_metrics_json_path", type=str, default="train_metrics.json")
    parser.add_argument("--saved_checkpoints_folder", type=str, default="saved_checkpoints")

    return parser.parse_args()


def main(args):
    device = get_device()
    is_cuda = device.type == 'cuda'
    print(f'Using device: {device}')
    print(args)

    train_set = MRSpectrogramDatasetLoader(
        noisy_dir=args.train_dataset_noisy_dir,
        clean_dir=args.train_dataset_clean_dir,
        segment_frames=args.segment_frames,
        hop_frames=args.hop_frames,
        pad_to_full_chunk=False
    )

    val_set = MRSpectrogramDatasetLoader(
        noisy_dir=args.val_dataset_noisy_dir,
        clean_dir=args.val_dataset_clean_dir,
        segment_frames=args.segment_frames,
        hop_frames=args.hop_frames,
        pad_to_full_chunk=False
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=is_cuda, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=is_cuda, num_workers=2)

    unet_args = get_unet_args()
    model = MultiStage_denoise(unet_args=unet_args)

    saved_checkpoint_counter = 0
    if os.path.exists(args.saved_checkpoints_folder):
        pth_ext = '.pth'
        latest_checkpoint_counter = -1
        latest_checkpoint_full_path = None

        for checkpoint_filename in os.listdir(args.saved_checkpoints_folder):
            if not checkpoint_filename.endswith(pth_ext):
                continue

            checkpoint_counter = int(checkpoint_filename[:-len(pth_ext)].split('_')[-1])
            if checkpoint_counter > latest_checkpoint_counter:
                latest_checkpoint_counter = checkpoint_counter
                latest_checkpoint_full_path = os.path.join(args.saved_checkpoints_folder, checkpoint_filename)

        if latest_checkpoint_full_path is not None:
            print(f'Continuing training from checkpoint {latest_checkpoint_full_path}')
            saved_checkpoint_counter = latest_checkpoint_counter + 1
            model.load_state_dict(torch.load(latest_checkpoint_full_path, map_location=device))
    elif args.fine_tune and os.path.exists(args.pretrained_path):
        print(f'Loading pre-trained weights from: {args.pretrained_path}')
        model.load_state_dict(torch.load(args.pretrained_path, map_location=device))

    model.to(device)
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        frozen_patience=args.frozen_patience,
        patience_after_all_unfrozen=args.patience_after_all_unfrozen,
        device=device,
        saved_checkpoints_folder=args.saved_checkpoints_folder,
        saved_metrics_json_path=args.saved_metrics_json_path,
        fine_tune=args.fine_tune,
        use_stage1_loss=args.use_stage1_loss,
        saved_checkpoint_counter=saved_checkpoint_counter
    )


def main_distributed(rank, world_size, args):
    try:
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo',
                                init_method='env://',
                                world_size=world_size,
                                rank=rank)

        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        is_cuda = device.type == 'cuda'
        torch.cuda.set_device(device) if is_cuda else None

        train_set = MRSpectrogramDatasetLoader(
            noisy_dir=args.train_dataset_noisy_dir,
            clean_dir=args.train_dataset_clean_dir,
            segment_frames=args.segment_frames,
            hop_frames=args.hop_frames,
            pad_to_full_chunk=False
        )

        val_set = MRSpectrogramDatasetLoader(
            noisy_dir=args.val_dataset_noisy_dir,
            clean_dir=args.val_dataset_clean_dir,
            segment_frames=args.segment_frames,
            hop_frames=args.hop_frames,
            pad_to_full_chunk=False
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                        num_replicas=world_size,
                                                                        rank=rank,
                                                                        shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set,
                                                                      num_replicas=world_size,
                                                                      rank=rank,
                                                                      shuffle=False)

        train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  pin_memory=is_cuda)
        val_loader = DataLoader(val_set,
                                batch_size=args.batch_size,
                                sampler=val_sampler,
                                pin_memory=is_cuda)

        unet_args = get_unet_args()
        model = MultiStage_denoise(unet_args=unet_args)

        if args.fine_tune and os.path.exists(args.pretrained_path):
            print(f'Loading pre-trained weights from: {args.pretrained_path}')
            model.load_state_dict(torch.load(args.pretrained_path, map_location=device, weights_only=True))

        model.to(device)
        model = DistributedDataParallel(model, device_ids=[rank] if is_cuda else None, find_unused_parameters=True)

        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            frozen_patience=args.frozen_patience,
            patience_after_all_unfrozen=args.patience_after_all_unfrozen,
            device=device,
            saved_checkpoints_folder=args.saved_checkpoints_folder,
            saved_metrics_json_path=args.saved_metrics_json_path,
            fine_tune=args.fine_tune,
            use_stage1_loss=args.use_stage1_loss
        )

        dist.destroy_process_group()
    except Exception as e:
        import traceback
        print(f"[RANK {rank}] Exception:")
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    train_args = parse_args()

    if train_args.distributed:
        w_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        mp.spawn(main_distributed, args=(w_size, train_args), nprocs=w_size)
    else:
        main(train_args)
