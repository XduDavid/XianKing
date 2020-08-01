import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import torchvision
import torchvision.transforms as transforms
import XianKingModel

transform = transforms.Compose(
     [transforms.Resize([128, 128]),
      transforms.ToTensor()])

trainset = torchvision.datasets.ImageFolder(root='./pic/train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.ImageFolder(root='./pic/test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=0)

net = XianKingModel.Net()
#net.load_state_dict(torch.load('XianKing_4w4a.pt'))        #在此基础上再次进行网络的训练

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
for epoch in range(200):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        inputs, labels = Variable(inputs), Variable(labels)
        labels = torch.unsqueeze(labels, dim=1)
        labels = torch.zeros(10, 10).scatter_(1, labels, 1)    #OneHot
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        outputs = outputs.reshape(10, 10)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.8f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0
print('Finished Training')

correct = 0  # 定义预测正确的图片数，初始化为0
total = 0  # 总共参与测试的图片数，也初始化为0
for data in testloader:  # 循环每一个batch
    images, labels = data
    outputs = net(Variable(images))  # 输入网络进行测试
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)  # 更新测试图片的数量
    predicted = predicted.reshape(10)
    correct += (predicted == labels).sum()  # 更新正确分类的图片的数量

print('Accuracy of the network on the 500 test images: %d %%' % (
        100 * correct / total))

print('===> Saving models...')

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(net.state_dict(), './checkpoint/XianKing_4w4a.pt')
