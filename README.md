# mamingDL

Deep learning library 

This library is based on 斎藤 康毅's dezero library(https://github.com/oreilly-japan/deep-learning-from-scratch-3)

The current version of this library only supports CPU. Support for GPU will be added in future updates. You can refer to the Python files under the "example" directory for usage instructions.

code usage:

```python
import numpy as np
import dezero
from dezero.models import MLP
import dezero.optimizers
import dezero.functions as F
from dezero.datasets import Spiral
from dezero.dataloaders import DataLoader
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
optimizer=dezero.optimizers.SGD(lr=lr).setup(model)

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

    with dezero.no_grad():
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
```

