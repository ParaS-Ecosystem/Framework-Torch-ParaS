from __future__ import print_function
import argparse
import time
import argparse
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import numpy as np




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    use_bn = True
    use_gp = False

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=0, bias=not self.use_bn)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=0)
        if self.use_gp:
            self.fc1 = nn.Linear(64, 16)
            self.fc2 = nn.Linear(16, 10)
        else:
            self.fc1 = nn.Linear(5 * 5 * 64, 128)
            self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        if self.use_gp:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        else:
            x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        output = self.fc2(x)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    start = time.time()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

    if str(device).startswith("paras") and hasattr(torch_paras, "synchronize"):
        torch_paras.synchronize()
    end = time.time()
    print("Epoch in %5.1fs" % (end - start))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data_dev = data.to(device)
            target_dev = target.to(device)
            output = model(data_dev)
            test_loss += F.cross_entropy(
                output, target_dev, reduction='sum').item()
            pred = output.to('cpu').argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model', default='conv', help='Model type conv, vit')
    parser.add_argument('--device', default='cpu',
                        help='cpu, paras:0 (paras CPU engine), or paras:1.. (paras GPU)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = args.device
    if device.startswith('paras'):
        global torch_paras
        import torch_paras
        n = torch_paras.device_count()
        print(f"[torch_paras] {n} device(s) visible "
              f"(device 0 = CPU engine, 1..{n-1} = GPU, if any)")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if device != 'cpu':
        extra_kwargs = {'num_workers': 1, 'shuffle': True}
        train_kwargs.update(extra_kwargs)
        test_kwargs.update(extra_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    print("Using device:", device)

    if args.model == 'conv':
        model = Net().to(device)
    elif args.model == 'vit':
        from vit_pytorch import ViT
        model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                    dim=64, depth=6, heads=8, mlp_dim=128)
        model.to(device)
    else:
        raise Exception("Unsupported model")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.to('cpu').state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
    print("Done")
