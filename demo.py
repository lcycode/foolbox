from __future__ import print_function

import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

import foolbox as fb
from foolbox.criteria import TargetedMisclassification

batch_size = 10
epoch = 1
learning_rate = 0.001
# 选择设备
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")


# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 训练模型
def train(model, optimizer, train_loader):
    for i in range(epoch):
        for j, (data, target) in tqdm(enumerate(train_loader)):
            data = data.to(device)
            target = target.to(device)
            logit = model(data)
            loss = F.nll_loss(logit, target)
            model.zero_grad()
            # 如下：因为其中的loss是单个tensor就不能用加上一个tensor的维度限制
            loss.backward()
            optimizer.step()
            if j % 1000 == 0:
                print('第{}个数据，loss值等于{}'.format(j, loss))


# 模型测试
def test(model, name, test_loader):
    correct_num = torch.tensor(0).to(device)
    for j, (data, target) in tqdm(enumerate(test_loader)):
        data = data.to(device)
        target = target.to(device)
        logit = model(data)
        pred = logit.max(1)[1]
        num = torch.sum(pred == target)
        correct_num = correct_num + num
    print(correct_num)
    print('\n{} correct rate is {}'.format(name, correct_num / 10000))


# 画图而已
def plot_clean_and_adver(adver_example, adver_target, clean_example, clean_target, attack_name):
    n_cols = 2
    n_rows = 5
    cnt = 1
    cnt1 = 1
    plt.figure(figsize=(4 * n_rows, 2 * n_cols))
    for i in range(n_cols):
        for j in range(n_rows):
            plt.subplot(n_cols, n_rows * 2, cnt1)
            plt.xticks([])
            plt.yticks([])
            if j == 0:
                plt.ylabel(attack_name, size=15)
            plt.title("{} -> {}".format(clean_target[cnt - 1], adver_target[cnt - 1]))
            plt.imshow(clean_example[cnt - 1].reshape(28, 28).to('cpu').detach().numpy(), cmap='gray')
            plt.subplot(n_cols, n_rows * 2, cnt1 + 1)
            plt.xticks([])
            plt.yticks([])
            # plt.title("{} -> {}".format(clean_target[cnt], adver_target[cnt]))
            plt.imshow(adver_example[cnt - 1].reshape(28, 28).to('cpu').detach().numpy(), cmap='gray')
            cnt = cnt + 1
            cnt1 = cnt1 + 2
    plt.show()
    print('\n')


# CW有目标攻击实现
def CW_target():
    start = time.time()
    criterion = TargetedMisclassification(torch.tensor([4] * batch_size, device=device))
    # 如下一行代码，定义攻击类型，其中攻击参数在此给定，参考github相应源码
    attack = fb.attacks.BoundaryAttack(steps=2)
    # 如下一行代码所示，实现有目标攻击。如下写法具有普适性
    raw, clipped, is_adv = attack(fmodel, images.to(device), epsilons=0.2, criterion=criterion)
    adver_target = torch.max(fmodel(raw), 1)[1]
    plot_clean_and_adver(raw, adver_target, images, labels, 'CW targeted')
    end = time.time()
    print('CW target running {} seconds using google colab GPU'.format((end - start)))


# CW无目标攻击实现
def CW_untarget():
    start = time.time()
    # criterion = TargetedMisclassification(torch.tensor([4]*batch_size,device = device))
    # 如下一行代码，定义攻击类型，其中攻击参数在此给定，参考github相应源码
    attack = fb.attacks.L2CarliniWagnerAttack()
    # 如下一行代码所示，实现无目标攻击。如下写法具有普适性
    raw, clipped, is_adv = attack(fmodel, images.to(device), labels.to(device), epsilons=0.2)
    adver_target = torch.max(fmodel(raw), 1)[1]
    plot_clean_and_adver(raw, adver_target, images, labels, 'CW untargeted')
    end = time.time()
    print('CW untargeted running {} seconds using google colab GPU'.format((end - start)))


def train_model():
    # 超参数设置

    # 加载mnist数据集
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=10, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=10, shuffle=True)

    # 初始化网络，并定义优化器
    simple_model = Net().to(device)
    optimizer1 = torch.optim.SGD(simple_model.parameters(), lr=learning_rate, momentum=0.9)
    print(simple_model)
    train(simple_model, optimizer1, train_loader=train_loader)
    # eval eval ，老子被你害惨了
    # 训练完模型后，要加上，固定DROPOUT层
    simple_model.eval()
    test(simple_model, 'simple model', test_loader=test_loader)

    torch.save(simple_model, "simple_model.pt")
    return simple_model


if not os.path.exists("./simple_model.pt"):
    simple_model = train_model()
else:
    simple_model = torch.load("./simple_model.pt")
    print(simple_model)

# 通过fb.PyTorchModel封装成的类，其fmodel使用与我们训练的simple_model基本一样
fmodel = fb.PyTorchModel(simple_model, bounds=(1, 1))
# 如下代码dataset可以选择cifar10,cifar100,imagenet,mnist等。其图像格式是channel_first
# 由于fmodel设置了bounds，如下代码fb.utils.samples获得的数据，会自动将其转化为bounds范围中
images, labels = fb.utils.samples(fmodel, dataset='mnist', batchsize=10)

print(images.shape)
print(labels.shape)
# CW_target()
CW_untarget()
