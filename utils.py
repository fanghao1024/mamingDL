import os
import subprocess
import numpy as np
import urllib.request

def _dot_var(v,verbose=False):
    dot_var='{} [label="{}",color=orange,style=filled]\n'
    name=v.name if v.name is not None else ''
    if verbose:
        if v.name is not None:
            name+=':'
        name+=str(v.shape)+' '+str(v.dtype)
    return dot_var.format(id(v),name)

def _dot_fun(f):
    dot_fun='{} [label="{}",color=lightblue,style=filled,shape=box]\n'
    ret=dot_fun.format(id(f),f.__class__.__name__)

    dot_edge='{}->{}\n'
    for x in f.inputs:
        ret+=dot_edge.format(id(x),id(f))
    for y in f.outputs:
        ret+=dot_edge.format(id(f),id(y()))
    return ret

def get_dot_graph(output,verbose=True):
    txt=''
    funcs=[]
    seen_funcs=set()

    def add_func(f):
        if f not in seen_funcs:
            funcs.append(f)
            seen_funcs.add(f)
    add_func(output.creator)
    txt+=_dot_var(output,verbose)
    while funcs:
        f=funcs.pop()
        txt+=_dot_fun(f)

        for x in f.inputs:
            txt+=_dot_var(x,verbose)

            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g{\n'+txt+'}'

def plot_dot_graph(output,verbose=True,to_file='sample.png'):
    dot_graph=get_dot_graph(output,verbose)

    dot_path=os.path.join(os.path.expanduser('~'),'.maming')
    if not os.path.exists(dot_path):
        os.mkdir(dot_path)
    tmp_dir=os.path.join(dot_path,'graph.dot')
    with open(tmp_dir,'w') as f:
        f.write(dot_graph)

    extension=os.path.splitext(to_file)[1][1:]
    cmd='dot {} -T {} -o {}'.format(tmp_dir,extension,to_file)
    subprocess.run(cmd,shell=True)

    try:
        from IPython import display
        return display.Image(filename=to_file)
    except:
        pass

def sum_to(x,shape):
    ndim=len(shape)
    lead=x.ndim-ndim
    lead_axis=tuple(range(lead))

    axis=tuple([i+lead for i,sx in enumerate(shape) if sx==1])
    y=x.sum(lead_axis+axis,keepdims=True)
    if lead>0:
        y=y.squeeze(lead_axis)
    return y

def reshape_sum_backward(gy,x_shape,axis,keepdims):
    ndim=len(x_shape)
    tupled_axis=axis
    if axis is None:
        tupled_axis=axis
    elif not isinstance(axis,tuple):
        tupled_axis=(axis,)

    if not (ndim==0 or axis is None or keepdims):
        shape=list(gy.shape)
        actual_axis=[a if a>0 else a+ndim for a in tupled_axis]
        for a in sorted(actual_axis):
            shape.insert(a,1)
    else:
        shape=gy.shape
    gy=gy.reshape(shape)
    return gy


def logsumexp(x,axis=1):
    m=x.max(axis=axis,keepdims=True)
    y=x-m
    y=np.exp(y)
    s=y.sum(axis=axis,keepdims=True)
    s=np.log(s)
    m+=s
    return m

def show_progress(block_num,block_size,total_size):
    bar_template="\r[{}]{:.3f}%"
    downloaded=block_num*block_size
    p=downloaded/total_size*100
    i=int(downloaded/total_size*30)
    if p>=100.0:p=100.0
    if i>=30:i=30
    bar='#'*i+'.'*(30-i)
    print(bar_template.format(bar,p),end='')

cache_dir=os.path.join(os.path.expanduser('~'),'.maming')
def get_file(url,file_name=None):
    if file_name is None:
        file_name=url[url.rfind('/')+1:]
    file_path=os.path.join(cache_dir,file_name)
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    if os.path.exists(file_path):
        os.remove(file_path)
    print('Downloading:'+file_name)
    try:
        urllib.request.urlretrieve(url,file_path,show_progress)
    except (Exception,KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print('Done')
    return file_path


def get_conv_outsize(input_size,kernel_size,stride,pad):
    return (input_size+pad*2-kernel_size)//stride+1

def pair(x):
    if isinstance(x,int):
        return (x,x)
    elif isinstance(x,tuple):
        assert len(x)==2
        return x
    else:
        raise ValueError