import numpy as np
import weakref
import contextlib

class Config:
    enable_backprop=True

@contextlib.contextmanager
def using_config(name,value):
    old_value=getattr(Config,name)
    setattr(Config,name,value)
    try:
        yield
    finally:
        setattr(Config,name,old_value)

def no_grad():
    return using_config('enable_backprop',False)

class Variable:
    __array_property__=200

    def __init__(self,data,name=None):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data=data
        self.name=name
        self.grad=None
        self.creator=None
        self.generation=0

    @property
    def shape(self):
        return self.data.shape
    @property
    def size(self):
        return self.data.size
    @property
    def ndim(self):
        return self.data.ndim
    @property
    def dtype(self):
        return self.data.dtype
    def __len__(self):
        return len(self.data)
    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        p=str(self.data).replace('\n','\n'+' '*9)
        return 'Variable('+p+')'

    def set_creator(self,f):
        self.creator=f
        self.generation=f.generation+1

    def cleargrad(self):
        self.grad=None

    def backward(self,retain_grad=False):
        if self.grad is None:
            self.grad=np.ones_like(self.data)

        funcs=[]
        seen_funcs=set()

        def add_func(f):
            if f not in seen_funcs:
                funcs.append(f)
                seen_funcs.add(f)
                funcs.sort(key=lambda x:x.generation)

        add_func(self.creator)

        while funcs:
            f=funcs.pop()
            gys=[output().grad for output in f.outputs]
            gxs=f.backward(*gys)
            if not isinstance(gxs,tuple):
                gxs=(gxs,)
            for x,gx in zip(f.inputs,gxs):
                if x.grad is None:
                    x.grad=gx
                else:
                    x.grad=x.grad+gx
                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for output in f.outputs:
                    output().grad=None
def as_variable(obj):
    if isinstance(obj,Variable):
        return obj
    return Variable(obj)
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self,*inputs):
        inputs=[as_variable(x) for x in inputs]
        xs=[x.data for x in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys,tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.outputs=[weakref.ref(output) for output in outputs]
            self.inputs=inputs

        return outputs if len(outputs)>1 else outputs[0]

    def forward(self,x):
        raise NotImplementedError()
    def backward(self,x):
        raise NotImplementedError()

class Add(Function):
    def forward(self,x0,x1):
        y=x0+x1
        return y
    def backward(self,gy):
        return gy,gy
def add(x0,x1):
    x1=as_array(x1)
    return Add()(x0,x1)

class Mul(Function):
    def forward(self,x0,x1):
        y=x0*x1
        return y
    def backward(self,gy):
        x0,x1=self.inputs[0].data,self.inputs[1].data
        return gy*x1,gy*x0
def mul(x0,x1):
    x1=as_array(x1)
    return Mul()(x0,x1)

class Sub(Function):
    def forward(self,x0,x1):
        y=x0-x1
        return y
    def backward(self,gy):
        return gy,-gy
def sub(x0,x1):
    x1=as_array(x1)
    return Sub()(x0,x1)
def rsub(x0,x1):
    x1=as_array(x1)
    return Sub()(x1,x0)

class Div(Function):
    def forward(self,x0,x1):
        y=x0/x1
        return y
    def backward(self,gy):
        x0,x1=self.inputs[0].data,self.inputs[1].data
        yx0=gy/x1
        yx1=gy*(-x0/x1**2)
        return yx0,yx1
def div(x0,x1):
    x1=as_array(x1)
    return Div()(x0,x1)
def rdiv(x0,x1):
    x1=as_array(x1)
    return Div()(x1,x0)

class Neg(Function):
    def forward(self,x):
        y=-x
        return y
    def backwar(self,gy):
        return -gy
def neg(x):
    return Neg()(x)

class Pow(Function):
    def __init__(self,c):
        self.c=c

    def forward(self,x):
        y=x**self.c
        return y

    def backward(self,gy):
        x=self.inputs[0].data
        gx=gy*x**(self.c-1)*self.c
        return gx
def pow(x,c):
    return Pow(c)(x)

def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__neg__ = neg
    Variable.__pow__ = pow