import numpy as np
import mamingDL
from mamingDL.models import MLP
import mamingDL.optimizers
import mamingDL.functions as F
from mamingDL.datasets import Spiral
from mamingDL.dataloaders import DataLoader
import matplotlib.pyplot as plt


max_epochs=300
batch_size=30
lr=1.0
hidden_size=10

train_dataset=Spiral(train=True)
test_dataset=Spiral(train=False)
train_dataloader=DataLoader(train_dataset,batch_size=batch_size)
test_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

model=MLP((hidden_size,3))
optimizer=mamingDL.optimizers.SGD(lr=lr).setup(model)

train_losses,train_accs,test_losses,test_accs=[],[],[],[]
for epoch in range(max_epochs):
    sum_acc=0
    sum_loss=0
    sum_num=0
    for x,t in train_dataloader:
        y=model(x)
        loss=F.softmax_cross_entropy(y,t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        acc=F.accuracy(y,t)
        sum_loss+=float(loss.data)*len(t)
        sum_acc+=float(acc.data)*len(t)
        sum_num+=len(t)
    print('epoch %d'%(epoch+1))
    print('train loss:{:.4f},accuracy:{:.4f}'.format(sum_loss/sum_num,sum_acc/sum_num))
    train_losses.append(sum_loss/sum_num)
    train_accs.append(sum_acc/sum_num)

    with mamingDL.no_grad():
        test_acc = 0
        test_sum=0
        test_loss=0
        for x,t in test_dataloader:
            y=model(x)
            loss=F.softmax_cross_entropy(y,t)

            test_loss+=float(loss.data)*len(t)
            acc=F.accuracy(y,t)
            test_acc+=float(acc.data)*len(t)
            test_sum+=len(t)
        print('test loss:{:.4f}, acc:{:.4f}'.format(test_loss/test_sum,test_acc/test_sum))
        test_losses.append(test_loss/test_sum)
        test_accs.append(test_acc/test_sum)

plt.plot(train_losses,label='train')
plt.plot(test_losses,label='test')
plt.legend()
plt.show()

plt.plot(train_accs)
plt.plot(test_accs)
plt.show()