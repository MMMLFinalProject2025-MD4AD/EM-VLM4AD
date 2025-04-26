from transformers import T5Tokenizer, TrainingArguments, Trainer
from torchvision import transforms
import json
import os
import time
from torch.utils.data import DataLoader
import torch
import argparse
from modules.multi_frame_dataset import MultiFrameDataset
from modules.multi_frame_model import print_trainable_parameters, DriveVLMT5
from modules.bevfusion_dataset import BevfusionDataset
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_checkpoint(model, optimizer, checkpoint_path, scheduler=None, load_orig_format=True, restart=False):

    if load_orig_format:
        model.load_state_dict(torch.load(checkpoint_path))
        return model, None, None, 0
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        raw_model_state = checkpoint['model_state_dict']
        if all(k.startswith("module.") for k in raw_model_state):
            model_state = {k.replace("module.", ""): v for k, v in raw_model_state.items()}
        else:
            model_state = raw_model_state
        model.load_state_dict(model_state)

        # Restore optimizer
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore scheduler if present
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Continue from last epoch
        if restart:
            start_epoch = 0
        else:
            start_epoch = checkpoint['epoch'] + 1
            lr = checkpoint.get('learning_rate', None)
            if lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        return model, optimizer, scheduler, start_epoch    

def save_model(model, model_name, output_dir, optimizer, epoch, config, scheduler=None):
    checkpoint = {
        'model_state_dict': model,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'learning_rate': optimizer.param_groups[0]['lr']
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()    
    path = os.path.join(output_dir, model_name + '.pth')
    torch.save(checkpoint, path)

'''
def save_model(model, model_name, output_dir):
    # Save the model into the designated folder
    path = os.path.join(output_dir, model_name + '.pth')
    torch.save(model.state_dict(), path)
'''


def val_model(dloader, val_model):
    val_model.eval()
    val_loss = 0

    for idx, (inputs, imgs, labels) in tqdm(enumerate(dloader), total=len(dloader)):
        outputs = val_model(inputs, imgs, labels)
        val_loss += outputs.loss.item()

    return val_loss / len(dloader)


def save_stats(train_loss, val_loss, epochs, lr, output_dir, losses, val_losses):
    stats_dict = {
        'losses': losses,
        'val_losses': val_losses,
        'min_train_loss': train_loss,
        'min_val_loss': val_loss,
        'epochs': epochs,
        'learning_rate': lr,
        'LM': 'T5-Base',
        'Image Embedding': 'Patch'
    }

    # Save stats into checkpoint
    with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
        json.dump(stats_dict, f)


def plot_loss(training_loss, val_loss, output_dir):
    num_epochs = len(training_loss)

    plt.plot(range(1, num_epochs + 1), training_loss, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Num epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss.png'))


def custom_train(train_loss, val_loss, best_model, epochs, model, optimizer, scheduler, train_dataloader, val_dataloader, config, output_dir, processor):
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)

    #if config.load_checkpoint and config.load_orig_format:
        #model, _, _, epochs = load_checkpoint(model, optimizer, config.checkpoint_file, scheduler, config.load_orig_format)
    #else:
        #model, optimizer, scheduler, epochs = load_checkpoint(model, optimizer, config.checkpoint_file, scheduler, config.load_orig_format)

    if not config.distributed or dist.get_rank() == 0:
        losses = []
        val_losses = []

    for epoch in range(epochs, config.epochs):
        if dist.get_rank() == 0:
            print('-------------------- EPOCH ' + str(epoch) + '/ TOTAL ' + str(config.epochs) + ' ---------------------')
        model.train()
        epoch_loss = 0

        #for step, (inputs, imgs, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        for step, (inputs, imgs, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            inputs = inputs.to(device)
            imgs = inputs.to(device)
            labels = labels.to(device)
            # Forward pass through model
            outputs = model(inputs, imgs, labels)

            # Calculate loss
            loss = outputs.loss
            epoch_loss += loss.item()

            if dist.get_rank() == 0 and step % config.checkpoint_frequency == 0:
                print('Loss: ' + str(loss.item()))
                with torch.no_grad():
                    outputs.logits[:,:, 32101:] = -float("inf")
                    hidden_states = outputs.logits.detach().cpu()
                    outputs_ids = torch.argmax(hidden_states, dim=-1)
                    text_outputs = [processor.decode(output.tolist(), skip_special_tokens=True) for output in outputs_ids]
                    text_questions = [processor.decode(q.detach().cpu().tolist(), skip_special_tokens=True) for q in inputs]
                    text_labels = [processor.decode(a.detach().cpu().tolist(), skip_special_tokens=True) for a in labels]

                print()
                print('Questions:')
                print(text_questions)
                print()
                print('Generated Answers:')
                print(text_outputs)
                print()
                print('Ground Truth Answers:')
                print(text_labels)

            # Back-propagate
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if config.distributed:
            dist.barrier()
        
        if not config.distributed or dist.get_rank() == 0:

            # Get train and val loss per batch
            epoch_train_loss = epoch_loss / len(train_dataloader)
            losses.append(epoch_train_loss)

            epoch_val_loss = val_model(val_dataloader, model)
            val_losses.append(epoch_val_loss)

            if not val_loss or min(epoch_val_loss, val_loss) == epoch_val_loss:
                val_loss = epoch_val_loss
                best_model = deepcopy(model.state_dict())
            if not train_loss or min(train_loss, epoch_train_loss) == epoch_train_loss:
                train_loss = epoch_train_loss

        # Adjust learning rate scheduler
        scheduler.step()

        if dist.get_rank() == 0:

            print('Training Loss: ' + str(epoch_train_loss))
            print('Validation Loss: ' + str(epoch_val_loss))
            print('---------------------------------------------')

        # Save model and stats for checkpoints
        if dist.get_rank() == 0:
            #save_model(best_model, 'latest_model', output_dir)
            print("Saving model... Keys:", list(best_model.keys())[:5])
            save_model(best_model, 'latest_model_saved', output_dir, optimizer, epoch, config, scheduler=scheduler)
        epochs += 1
        if dist.get_rank() == 0:
            save_stats(train_loss, val_loss, epochs, scheduler.get_last_lr()[0], output_dir, losses, val_losses)

    # Save the model and plot the loss
    if dist.get_rank() == 0:
        plot_loss(losses, val_losses, output_dir)
    return train_loss, val_loss


def train(config):
    ##timestr = time.strftime("%Y%m%d-%H%M%S")
    #output_dir = os.path.join('multi_frame_results', timestr)
    output_dir = config.output_dir
    print(f"output_dir = {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Load processors and models
    model = DriveVLMT5(config)
    model = model.to(device)
    if dist.get_rank() == 0:
        print_trainable_parameters(model)

    if config.lm == 'T5-Base':
        processor = T5Tokenizer.from_pretrained('google-t5/t5-base')
    else:
        processor = T5Tokenizer.from_pretrained('google-t5/t5-large')

    processor.add_tokens('<')
    processor.model_max_length = 512  # for msg: 'Token indices sequence length is longer than the specified maximum sequence length for this model (583 > 512). Running this sequence through the model will result in indexing errors'

    if config.feat == 'image':
        DatasetClass = MultiFrameDataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])
    elif config.feat == 'bevfusion':
        DatasetClass = BevfusionDataset
        transform = None
    else:
        raise ValueError(f"Unknown feat type: {config.feat}")

    train_dset = DatasetClass(
        input_file=os.path.join('data', 'multi_frame',
                                'multi_frame_train.json'),
        tokenizer=processor,
        transform=transform
    )
    val_dset = DatasetClass(
        input_file=os.path.join('data', 'multi_frame',
                                'multi_frame_val.json'),
        tokenizer=processor,
        transform=transform
    )
    test_dset = DatasetClass(
        input_file=os.path.join('data', 'multi_frame',
                                'multi_frame_test.json'),
        tokenizer=processor,
        transform=transform
    )

    train_sampler = DistributedSampler(train_dset) if config.distributed else None
    val_sampler = DistributedSampler(val_dset, shuffle=False) if config.distributed else None
    test_sampler = DistributedSampler(test_dset, shuffle=False) if config.distributed else None

    train_dataloader = DataLoader(
        train_dset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        collate_fn=train_dset.collate_fn
    )
    val_dataloader = DataLoader(
        val_dset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=train_dset.collate_fn
    )
    test_dataloader = DataLoader(
        test_dset,
        batch_size=config.batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=test_dset.collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)
    epochs = 0

    if config.load_checkpoint and config.load_orig_format:
        model, _, _, epochs = load_checkpoint(model, optimizer, config.checkpoint_file, scheduler, config.load_orig_format)
    elif config.load_checkpoint:
        model, optimizer, scheduler, epochs = load_checkpoint(model, optimizer, config.checkpoint_file, scheduler, config.load_orig_format, config.restart)

    if dist.is_initialized():
        dist.barrier()        

    # Initialize DistributedDataParallel
    model = DDP(model, device_ids=[config.local_rank])

    # Train the model
    min_train_loss, min_val_loss = custom_train(None, None, None, epochs, model, optimizer, scheduler, train_dataloader, val_dataloader, config, output_dir, processor)
    return min_train_loss, min_val_loss


def params():

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", default=1e-4, type=float,
                        help="Model learning rate starting point, default is 1e-4.")
    parser.add_argument("--batch-size", default=4, type=int,
                        help="Batch size per GPU/CPU for training and evaluation, defaults to 4.")
    parser.add_argument("--weight-decay", default=0.05, type=float,
                        help="L2 Regularization, default is 0.05")
    parser.add_argument("--epochs", default=15, type=int,
                        help="Number of epochs to train for, default is 15")
    parser.add_argument("--hf-train", action='store_true',
                        help="Whether to use HuggingFace default training or custom training loop")
    parser.add_argument('--gpa-hidden-size', default=128, type=int, help='Hidden dimension for Gated Pooling Attention, '
                                                                         'default is 128')
    parser.add_argument('--freeze-lm', action='store_true', help='Freeze LM during training')
    parser.add_argument('--lm', default='T5-Base', choices=['T5-Base', 'T5-Large'], type=str, help='Backbone LM to use, '
                                                                                        'use \'T5-Base\' for T5-Medium')
    parser.add_argument('--checkpoint-frequency', default=500, type=int, help='Frequency of showing example outputs')
    parser.add_argument('--lora', action='store_true', help='Perform LoRA finetuning, recommend if '
                                                            'using T5-Large backbone LM')
    parser.add_argument('--lora-dim', default=64, type=int, help='LoRA dimension')
    parser.add_argument('--lora-alpha', default=32, type=int, help='LoRA alpha')
    parser.add_argument('--lora-dropout', default=0.05, type=float, help='LoRA dropout')
    parser.add_argument('--num-workers', default=8, type=int, help='# of Workers used by Dataloader')
    parser.add_argument('--load-checkpoint', action='store_true', help='Whether to load a checkpoint from '
                                                                       'multi_frame_results folder')
    parser.add_argument('--checkpoint-file', default='./multi_frame_results/T5-Base/latest_model.pth', type=str, help='The checkpoint to load from multi_frame_results directory')
    parser.add_argument('--feat', default='bevfusion', choices=['bevfusion', 'image'], type=str, help='Feature used')
    parser.add_argument('--mask-img', action='store_true', help='Mask out the img embedding.')
    parser.add_argument('--restart', action='store_true', help='Restart epoch from 0.')
    parser.add_argument('--load-orig-format', action='store_true', help='Load the checkpoint format from the original checkpoint of the author.')
    parser.add_argument('--output-dir', default='./multi_frame_results/outputs', type=str, help='The output directory for saveing checkpoint and stats')

    args = parser.parse_args()
    return args


def init_distributed_mode(config):
    """Initialize the distributed mode for DDP."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        config.rank = int(os.environ['RANK'])
        config.world_size = int(os.environ['WORLD_SIZE'])
        config.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        config.rank = int(os.environ['SLURM_PROCID'])
        config.local_rank = config.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        config.distributed = False
        return

    config.distributed = True

    torch.cuda.set_device(config.local_rank)
    print(f'| distributed init (rank {config.rank}): env://', flush=True)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=config.world_size,
        rank=config.rank
    )
    dist.barrier()

def main():
    config = params()
    init_distributed_mode(config)
    train(config)
    


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
