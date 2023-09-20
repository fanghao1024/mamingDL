import numpy as np
from mamingDL import cuda
from mamingDL.core import Function,as_variable
from mamingDL.utils import pair,get_conv_outsize
from mamingDL.functions import linear,broadcast_to

class Im2col(Function):
    def __init__(self,kernel_size,stride,pad,to_matrix):
        super(Im2col, self).__init__()
        self.input_shape=None
        self.kernel_size=kernel_size
        self.stride=stride
        self.pad=pad
        self.to_matrix=to_matrix

    def forward(self,x):
        self.input_shape=x.shape
        y=im2col_array(x,self.kernel_size,self.stride,self.pad,self.to_matrix)
        return y

    def backward(self,gy):
        gx=col2im(gy,self.input_shape,self.kernel_size,self.stride,self.pad,self.to_matrix)
        return gx

def im2col(x,kernel_size,stride=1,pad=0,to_matrix=0):
    y=Im2col(kernel_size,stride,pad,to_matrix)(x)
    return y

class Col2im(Function):
    def __init__(self,input_shape,kernel_size,stride,pad,to_matrix):
        super(Col2im, self).__init__()
        self.input_shape=input_shape
        self.kernel_size=kernel_size
        self.stride=stride
        self.pad=pad
        self.to_matrix=to_matrix

    def forward(self,x):
        y=col2im_array(x,self.input_shape,self.kernel_size,self.stride,self.pad,self.to_matrix)
        return y

    def backward(self,gy):
        gx=im2col(gy,self.kernel_size,self.stride,self.pad,self.to_matrix)
        return gx

def col2im(x,input_shape,kernel_size,stride=1,pad=0,to_matrix=True):
    return Col2im(input_shape,kernel_size,stride,pad,to_matrix)(x)

def im2col_array(img,kernel_size,stride,pad,to_matrix=True):
    N,C,H,W=img.shape
    KH,KW=pair(kernel_size)
    SH,SW=pair(stride)
    PH,PW=pair(pad)
    OH=get_conv_outsize(H,KH,SH,PH)
    OW=get_conv_outsize(W,KW,SW,PW)

    xp=cuda.get_array_module(img)
    if xp!=np:
        col=_im2col_gpu(img,kernel_size,stride,pad)
    else:
        '''
        (0, 0)：表示在第一个维度上不进行填充，即不改变原始数组的大小。
        (0, 0)：表示在第二个维度上不进行填充，即不改变原始数组的大小。
        (PH, PH + SH - 1)：表示在第三个维度上进行填充，分别在数组的开头和结尾添加。
        (PW, PW + SW - 1)：表示在第四个维度上进行填充，分别在数组的开头和结尾添加。
        '''
        img=np.pad(img,((0,0),(0,0),(PH,PH+SH-1),(PW,PW+SW-1)),mode='constant',constant_values=(0,))
        col=np.ndarray((N,C,KH,KW,OH,OW),dtype=img.dtype)

        for j in range(KH):
            j_lim=j+SH*OH
            for i in range(KW):
                i_lim=i+SW*OW
                col[:,:,j,i,:,:]=img[:,:,j:j_lim:SH,i:i_lim:SW]
    if to_matrix:
        col=col.transpose((0,4,5,1,2,3)).reshape(N*OH*OW,-1)
    return col

def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    xp = cuda.get_array_module(col)
    if xp != np:
        img = _col2im_gpu(col, SH, SW, PH, PW, H, W)
        return img
    else:
        img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1),
                       dtype=col.dtype)
        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
        return img[:, :, PH:H + PH, PW:W + PW]

def _im2col_gpu(img, kernel_size, stride, pad):
    """im2col function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, h, w = img.shape
    kh, kw = pair(kernel_size)
    sy, sx = pair(stride)
    ph, pw = pair(pad)
    out_h = get_conv_outsize(h, kh, sy, ph)
    out_w = get_conv_outsize(w, kw, sx, pw)
    dy, dx = 1, 1
    col = cuda.cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    cuda.cupy.ElementwiseKernel(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)

    return col

def _col2im_gpu(col, sy, sx, ph, pw, h, w):
    """col2im function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, kh, kw, out_h, out_w = col.shape
    dx, dy = 1, 1
    img = cuda.cupy.empty((n, c, h, w), dtype=col.dtype)

    cuda.cupy.ElementwiseKernel(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img

def conv2d_simple(x,W,b=None,stride=1,pad=0):
    x,W=as_variable(x),as_variable(W)

    Weight=W
    N,C,H,W=x.shape
    OC,C,KH,KW=Weight.shape
    SH,SW=pair(stride)
    PH,PW=pair(pad)
    OH=get_conv_outsize(H,KH,SH,PH)
    OW=get_conv_outsize(W,KW,SW,PW)

    col=im2col(x,(KH,KW),stride,pad,to_matrix=True)
    Weight=Weight.reshape(OC,-1).transpose()
    t=linear(col,Weight,b)
    y=t.reshape(N,OH,OW,OC).transpose(0,3,1,2)
    return y

def pooling_simple(x,kernel_size,stride=1,pad=0):
    x=as_variable(x)

    N,C,H,W=x.shape
    KH,KW=pair(kernel_size)
    SH,SW=pair(stride)
    PH,PW=pair(pad)
    OH=get_conv_outsize(H,KH,SH,PH)
    OW=get_conv_outsize(W,KW,SW,PW)

    col=im2col(x,kernel_size,stride,pad,to_matrix=True)
    col=col.reshape(-1,KH*KW)
    y=col.max(axis=1)
    y=y.reshape(N,OH,OW,C).transpose(0,3,1,2)
    return y
