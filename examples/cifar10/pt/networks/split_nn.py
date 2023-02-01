import numpy as np
import torch
from pt.networks.cifar10_nets import ModerateCNN
from pt.utils.cifar10_dataset import CIFAR10SplitNN
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


# TODO: maybe only use the part net that is being used
#  rather than inheriting the full net
class SplitNN(ModerateCNN):
    def __init__(self, split_id):
        super().__init__()
        if split_id not in [0, 1]:
            raise ValueError(f"Only supports split_id '0' or '1' but was {self.split_id}")
        self.split_id = split_id

        if self.split_id == 0:
            self.split_forward = self.conv_layer
        elif self.split_id == 1:
            self.split_forward = self.fc_layer
        else:
            raise ValueError(f"Expected split_id to be '0' or '1' but was {self.split_id}")

    def forward(self, x):
        x = self.split_forward(x)
        return x

    def get_split_id(self):
        return self.split_id


""" TESTING """


def print_grads(net):
    for name, param in net.named_parameters():
        if param.grad is not None:
            print(name, "grad", param.grad.shape, torch.sum(param.grad).item())
        else:
            print(name, "grad", None)


def test_splitnn():  # TODO: move to unit testing
    """Test SplitNN"""

    lr = 1e-2
    epoch_max = 20
    bs = 64

    train_size = 50000

    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net1 = SplitNN(split_id=0).to(device)
    net2 = SplitNN(split_id=1).to(device)

    optim1 = optim.SGD(net1.parameters(), lr=lr, momentum=0.9)
    optim2 = optim.SGD(net2.parameters(), lr=lr, momentum=0.9)

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )
    transform_valid = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root="/tmp/cifar10_vertical",
        train=True,
        download=True,
        transform=transform_train,
    )
    valid_dataset = datasets.CIFAR10(
        root="/tmp/cifar10_vertical",
        train=False,
        download=False,
        transform=transform_valid,
    )

    train_image_dataset = CIFAR10SplitNN(
        root="/tmp/cifar10_vertical", train=True, download=True, transform=transform_train, returns="image"
    )
    train_label_dataset = CIFAR10SplitNN(
        root="/tmp/cifar10_vertical", train=True, download=True, transform=transform_train, returns="label"
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=2)

    def valid(model1, model2, data_loader, device):
        model1.eval()
        model2.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for i, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model1(inputs)
                outputs = model2(outputs)
                _, pred_label = torch.max(outputs.data, 1)

                total += inputs.data.size()[0]
                correct += (pred_label == labels.data).sum().item()
            metric = correct / float(total)
        return metric

    def train(inputs, targets, debug=False):
        # See also
        # https://github.com/Koukyosyumei/Attack_SplitNN/blob/main/src/attacksplitnn/splitnn/model.py

        """Compute on site-1"""
        net1.train()
        optim1.zero_grad()

        x = net1.forward(inputs)  # keep on site-1
        x_sent = x.detach().requires_grad_()
        # send to site-2

        """ Compute on site-2 """
        net2.train()
        optim2.zero_grad()

        pred = net2.forward(x_sent)

        loss = criterion(pred, targets)

        loss.backward()
        optim2.step()

        return_grad = x_sent.grad

        if debug:
            print("return_grad", return_grad.shape, torch.sum(return_grad))

            print("====== net2 grad: ======")
            print_grads(net2)

        # send gradients to site-1

        """ Compute on site-1 """
        x.backward(gradient=return_grad)
        optim1.step()

        if debug:
            print("====== net1 grad: ======")
            print_grads(net1)

        return loss.item()

    # main training loop
    writer = SummaryWriter("./")
    # epoch_len = len(train_loader)
    epoch_len = int(train_size / bs)
    for e in range(epoch_max):
        epoch_loss = 0
        # for i, (inputs, targets) in enumerate(train_loader):
        # epoch_loss += train(inputs=inputs.to(device), targets=targets.to(device))

        for i in range(epoch_len):
            batch_indices = np.random.randint(0, train_size - 1, bs)
            inputs = train_image_dataset.get_batch(batch_indices)
            targets = train_label_dataset.get_batch(batch_indices)
            loss = train(inputs=inputs.to(device), targets=targets.to(device))
            epoch_loss += loss
            writer.add_scalar("loss", loss, e * epoch_len + i)

        train_acc = valid(net1, net2, train_loader, device)
        val_acc = valid(net1, net2, valid_loader, device)
        print(
            f"Epoch {e+1}/{epoch_max}. loss: {epoch_loss/epoch_len:.4f}, "
            f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}"
        )
        writer.add_scalar("train_acc", train_acc, e)
        writer.add_scalar("val_acc", val_acc, e)


if __name__ == "__main__":
    test_splitnn()
