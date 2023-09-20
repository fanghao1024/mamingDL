import mamingDL
from mamingDL.core import Variable,Function,as_variable,as_array
from mamingDL import utils,cuda
import numpy as np

class Sin(Function):
    def forward(self,x):
        xp=cuda.get_array_module(x)
        y=xp.sin(x)
        return y
    def backward(self,gy):
        x,=self.inputs
        gx=gy*cos(x)
        return gx
def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self,x):
        xp = cuda.get_array_module(x)
        y=xp.cos(x)
        return y
    def backward(self,gy):
        x,=self.inputs
        gx=-gy*sin(x)
        return gx
def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self,x):
        xp = cuda.get_array_module(x)
        y=xp.tanh(x)
        return y
    def backward(self,gy):
        y=self.outputs[0]()
        gx=gy*(1-y*y)
        return gx
def tanh(x):
    return Tanh()(x)

class Exp(Function):
    def forward(self,x):
        xp = cuda.get_array_module(x)
        y=xp.exp(x)
        return y
    def backward(self,gy):
        y=self.outputs[0]()
        gx=gy*y
        return gx
def exp(x):
    return Exp()(x)

class Reshape(Function):
    def __init__(self,shape):
        self.shape=shape

    def forward(self,x):
        self.x_shape=x.shape
        y=x.reshape(self.shape)
        return y
    def backward(self,gy):
        return reshape(gy,self.x_shape)

def reshape(x,shape):
    if x.shape==shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Function):
    def __init__(self,axes=None):
        self.axes=axes

    def forward(self,x):
        y=x.transpose(self.axes)
        return y
    def backward(self,gy):
        if self.axes is None:
            return transpose(gy)
        axes_len=len(self.axes)
        inv_axes=tuple(np.argsort([ax % axes_len for ax in self.axes]))
        gx=transpose(gy)
        return gx

def transpose(x,axes=None):
    return Transpose(axes)(x)

class BroadcastTo(Function):
    def __init__(self,shape):
        self.shape=shape
    def forward(self,x):
        self.x_shape=x.shape
        xp = cuda.get_array_module(x)
        y=xp.broadcast_to(x,self.shape)
        return y
    def backward(self,gy):
        gx=sum_to(gy,self.x_shape)
        return gx
def broadcast_to(x,shape):
    if x.shape==shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self,shape):
        self.shape=shape
    def forward(self,x):
        self.x_shape=x.shape
        y=utils.sum_to(x,self.shape)
        return y
    def backward(self,gy):
        gx=broadcast_to(gy,self.x_shape)
        return gx
def sum_to(x,shape):
    if x.shape==shape:
        return as_variable(x)
    return SumTo(shape)(x)

class Sum(Function):
    def __init__(self,axis,keepdims):
        self.axis=axis
        self.keepdims=keepdims
    def forward(self,x):
        self.x_shape=x.shape
        y=x.sum(axis=self.axis,keepdims=self.keepdims)
        return y
    def backward(self,gy):
        gy=utils.reshape_sum_backward(gy,self.x_shape,self.axis,self.keepdims)
        gx=broadcast_to(gy,self.x_shape)
        return gx
def sum(x,axis=None,keepdims=False):
    return Sum(axis,keepdims)(x)

class Matmul(Function):
    def forward(self,x,W):
        y=x.dot(W)
        return y
    def backward(self,gy):
        x,W=self.inputs
        dx=matmul(gy,W.T)
        dW=matmul(x.T,gy)
        return dx,dW
def matmul(x,W):
    return Matmul()(x,W)


class MeanSquareError(Function):
    def forward(self,x0,x1):
        diff=x0-x1
        return (diff**2).sum()/len(diff)
    def backward(self,gy):
        x0,x1=self.inputs
        diff=x0-x1
        gx0=gy*diff*2./len(diff)
        gx1=-gx0
        return gx0,gx1
def mean_square_error(x0,x1):
    return MeanSquareError()(x0,x1)

def linear_simple(x,W,b=None):
    t=Matmul(x,W)
    if b is None:
        return t
    y=t+b
    t.data=None
    return y

class Linear(Function):
    def forward(self,x,W,b):
        y=x.dot(W)
        if b is not None:
            y+=b
        return y
    def backward(self,gy):
        x,W,b=self.inputs
        gb=None if b.data is None else sum_to(gy,b.shape)
        gx=matmul(gy,W.T)
        gW=matmul(x.T,gy)
        return gx,gW,gb
def linear(x,W,b):
    return Linear()(x,W,b)

def sigmoid_simple(x):
    x=as_variable(x)
    return 1./(1+exp(-x))

class Sigmoid(Function):
    def forward(self,x):
        xp = cuda.get_array_module(x)
        y=xp.tanh(x*0.5)*0.5+0.5
        return y
    def backward(self,gy):
        y=self.outputs[0]()
        gx=gy*y*(1-y)
        return gx
def sigmoid(x):
    return Sigmoid()(x)

class ReLU(Function):
    def forward(self,x):
        xp = cuda.get_array_module(x)
        y=xp.maximum(x,0.0)
        return y
    def backward(self,gy):
        x,=self.inputs
        mask=x.data>0
        gx=gy*mask
        return gx
def relu(x):
    return ReLU()(x)

class GetItem(Function):
    def __init__(self,slices):
        self.slices=slices
    def forward(self,x):
        y=x[self.slices]
        return y
    def backward(self,gy):
        x,=self.inputs
        f=GetItemGrad(self.slices,x.shape)
        return f(gy)

class GetItemGrad(Function):
    def __init__(self,slices,shape):
        self.slices=slices
        self.shape=shape
    def forward(self,gy):
        gx=np.zeros(self.shape,dtype=gy.dtype)
        np.add.at(gx,self.slices,gy)
        return gx
    def backward(self,ggx):
        return get_item(ggx,self.slices)


class Clip(Function):
    def __init__(self,x_min,x_max):
        self.x_min=x_min
        self.x_max=x_max
    def forward(self,x):
        xp = cuda.get_array_module(x)
        y=xp.clip(x,self.x_min,self.x_max)
        return y
    def backward(self,gy):
        x,=self.inputs
        mask=(x.data>=self.x_min)*(x.data<=self.x_max)
        gx=gy*mask
        return gx
def clip(x,x_min,x_max):
    return Clip(x_min,x_max)(x)

class Log(Function):
    def forward(self,x):
        xp = cuda.get_array_module(x)
        y=xp.log(x)
        return y
    def bacwkard(self,gy):
        x,=self.inputs
        x=np.clip(x,1e-15,np.inf)
        gx=gy/x
        return gx
def log(x):
    return Log()(x)

def get_item(x,shape):
    f=GetItem(shape)
    return f(x)

def softmax_simple(x):
    x=as_variable(x)
    y=exp(x)
    y_sum=sum(y,axis=1,keepdims=True)
    return y/y_sum

def softmax_cross_entropy_simple(x,t):
    x,t=as_variable(x),as_variable(t)
    N=x.shape[0]
    p=softmax_simple(x)
    p=clip(p,1e-15,1.)
    log_p=log(p)
    tlog_p=log_p[np.arange(N),t.data]
    y=-1*sum(tlog_p)/N
    return y

class Softmax(Function):
    def __init__(self,axis=1):
        self.axis=1
    def forward(self,x):
        xp = cuda.get_array_module(x)
        y=x-x.max(axis=self.axis,keepdims=True)
        y=xp.exp(y)
        y/=y.sum(axis=self.axis,keepdims=True)
        return y
    def backward(self,gy):
        y=self.outputs[0]()
        gx=y*gy
        sumdx=gx.sum(axis=self.axis,keepdims=True)
        gx-=y*sumdx
        return gx
def softmax(x,axis=1):
    return Softmax(axis)(x)

class SoftmaxCrossEntropy(Function):
    def forward(self,x,t):
        N=x.shape[0]
        log_z=utils.logsumexp(x,axis=1)
        log_p=x-log_z
        log_p=log_p[np.arange(N),t.ravel()]
        y=-log_p.sum()/np.float32(N)
        return y
    def backward(self,gy):
        x,t=self.inputs
        N,CLS_NUM=x.shape

        gy*=1/N
        y=softmax(x)
        xp=cuda.get_array_module(t.data)
        t_onehot=xp.eye(CLS_NUM,dtype=t.dtype)[t.data]
        y=(y-t_onehot)*gy
        return y
def softmax_cross_entropy(x,t):
    return SoftmaxCrossEntropy()(x,t)


def accuracy(y,t):
    y,t=as_variable(y),as_variable(t)
    pred=y.data.argmax(axis=1).reshape(t.shape)
    result=(pred==t.data)
    acc=result.mean()
    return Variable(as_array(acc))

def dropout(x,dropout_rate=0.5):
    x=as_variable(x)

    if mamingDL.Config.train:
        xp=cuda.get_array_module(x)
        mask=xp.random.rand(*x.shape)>dropout_rate
        scale=xp.array(1-dropout_rate).astype(x.dtype)
        y=x*mask/scale
        return y
    else:
        return x




from mamingDL.function_conv import im2col
from mamingDL.function_conv import col2im
from mamingDL.function_conv import conv2d_simple

from mamingDL.core import add
from mamingDL.core import sub
from mamingDL.core import rsub
from mamingDL.core import mul
from mamingDL.core import div
from mamingDL.core import neg
from mamingDL.core import neg
from mamingDL.core import pow



