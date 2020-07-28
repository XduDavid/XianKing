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
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(root='./pic/train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.ImageFolder(root='./pic/test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)


net = XianKingModel.Net()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        inputs, labels = Variable(inputs), Variable(labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')

correct = 0  # 定义预测正确的图片数，初始化为0
total = 0  # 总共参与测试的图片数，也初始化为0
for data in testloader:  # 循环每一个batch
    images, labels = data
    outputs = net(Variable(images))  # 输入网络进行测试
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)  # 更新测试图片的数量
    correct += (predicted == labels).sum()  # 更新正确分类的图片的数量

print('Accuracy of the network on the 500 test images: %d %%' % (
        100 * correct / total))

#for parameters in net.parameters():
#   print(parameters)

print('===> Saving models...')
# state = {
#     'state': net.state_dict(),
#     'epoch': epoch                   # 将epoch一并保存
# }
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(net.state_dict(), './checkpoint/XianKing_4w4a.pt')
