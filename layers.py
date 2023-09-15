from mamingDL.core import Parameter
import weakref
import numpy as np
import mamingDL.functions as F

class Layer:
    def __init__(self):
        self._params=set()

    def __setattr__(self, name, value):
        if isinstance(value,(Parameter,Layer)):
            self._params.add(name)
        super().__setattr__(name,value)

    def __call__(self, *inputs):
        outputs=self.forward(*inputs)
        if not isinstance(outputs,tuple):
            outputs=(outputs,)
        self.inputs=[weakref.ref(x) for x in inputs]
        self.outputs=[weakref.ref(y) for y in outputs]
        return outputs if len(outputs)>1 else outputs[0]
    def forward(self,x):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj=self.__dict__[name]
            if isinstance(obj,Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for params in self.params():
            params.cleargrad()

class Linear(Layer):
    def __init__(self,out_size,no_bias=False,dtype=np.float32,in_size=None):
        super(Linear, self).__init__()

        self.in_size=in_size
        self.out_size=out_size
        self.dtype=dtype

        self.W=Parameter(None,name='W')
        if in_size is not None:
            self._initW()

        if no_bias:
            self.b=None
        else:
            self.b=Parameter(np.zeros(out_size,dtype=dtype),name='b')
    def _initW(self):
        I,O=self.in_size,self.out_size
        W_data=np.random.randn(I,O).astype(self.dtype)*np.sqrt(1/I)
        self.W.data=W_data
    def forward(self,x):
        if self.in_size is None:
            self.in_size=x.shape[1]
            self._initW()
        y=F.linear(x,self.W,self.b)
        return y