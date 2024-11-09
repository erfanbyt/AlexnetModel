from dataset import get_train_val_loader, get_test_loader
from model import AlexNet
import torch
import torch.nn as nn

# Hyper-parameters
num_classes = 10
num_epoches = 1
batch_size = 64
learning_rate = 0.005

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining the model
model = AlexNet(num_classes=num_classes).to(device)

# loading the dataset
train_loader, valid_loader = get_train_val_loader(data_dir="./data", batch_size=64, augment=False, random_seed=1)
test_loader = get_test_loader(data_dir='./data', batch_size=64)

# loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

# train the model --> return total number of the batches in the train dataset
total_steps= len(train_loader)
print(total_steps)
for epoch in range(num_epoches):
    print(epoch)
    for i, (images, labels) in enumerate(train_loader):
        print(i)
        # move the tensors to the device
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epoches, i+1, total_steps, loss.item()))
    
    # validation
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

