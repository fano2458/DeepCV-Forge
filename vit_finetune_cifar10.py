import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from transformers import ViTForImageClassification

from tqdm import tqdm
import time
import math
import inspect

from src.vit import ViT, ViTConfig


num_epoch = 10
max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
# total_batch_size = 256
batch_size = 128

# assert total_batch_size % batch_size == 0, 'make sure that total_batch_size is divisible by batch_size'
# grad_accum_steps = total_batch_size / batch_size


def configure_optimizer(weight_decay, learning_rate, device, model):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params':decay_params, 'weight_decay':weight_decay},
        {'params':nodecay_params, 'weight_decay':0.0}
    ]

    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and 'cuda' in device
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer


def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)



def main():
    torch.set_float32_matmul_precision('high')  # to enable TF32 precision

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),  # Similar to HorizontalFlip
        torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5),  # Approximates RandomBrightnessContrast
        torchvision.transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(1-0.05, 1+0.05)),  # Similar to ShiftScaleRotate
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.2154, 0.2241)),
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.2154, 0.2241)),
    ])

    # train_transform = torch.compile(train_transform)  ????
    # test_transform = torch.compile(test_transform)    #TODO consider torch.jit.script

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_train = torchvision.datasets.CIFAR10(root='/home/fano/Desktop/DeepCV-Forge/data', train=True, download=True, transform=train_transform)
    dataset_test = torchvision.datasets.CIFAR10(root='/home/fano/Desktop/DeepCV-Forge/data', train=False, download=True, transform=test_transform)

    train_set, val_set = random_split(dataset_train, [40000, 10000])

    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True) # should be determined
    dataloader_valid = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = ViT.from_pretrained('facebook/deit-tiny-patch16-224', n_class=10).to(device) # load pretrained model
    # model.requires_grad_(False)                                                          # freeze all parameters
    # model.classifier.requires_grad_(True)                                                # set last layer to be trainable
    # model = Net().to(device)

    # summary(model, input_size=(1, 3, 32, 32), 
    #         col_names=['input_size', 'output_size', 'num_params', 'trainable'], 
    #         device=device, col_width=20, depth=4)
    
    # model = ViT(ViTConfig(n_class=10, patch_size=32)).to(device).train()
    # print(model.parameters())
    # model = torch.compile(model)
    # model.train()
    # optimizer = configure_optimizer(model=model, weight_decay=0.01,
    #                                 learning_rate=max_lr, device=device)
    # model = ViTForImageClassification.from_pretrained("facebook/deit-tiny-patch16-224", num_labels=10, ignore_mismatched_sizes=True).to(device)
    # model = ViT(ViTConfig(n_class=10)).to(device)
    model = torch.compile(model)
    # model.classifier = nn.Linear(192, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss().cuda()

    for step in range(10):
        # train
        train_loss = 0.0
        for data, target in tqdm(dataloader_train):
            data, target = data.to(device), target.to(device)

            # print(target)
            # print(model(data))

            # import sys; sys.exit(0)
            # print(data.shape)
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                out = model(data)

            loss = criterion(out, target)
            loss.backward()

            train_loss = loss.detach()

            # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # lr = get_lr(step) # should change with batch
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr

            optimizer.step()
            torch.cuda.synchronize()

        print(f"{train_loss:2f}")

        # eval

        # test



if __name__ == "__main__":
    main()
