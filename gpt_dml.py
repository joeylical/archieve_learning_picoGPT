import torch as th
import torch_directml
import os
import sys
import time
import json
import pickle
from functools import wraps
from encoder import get_encoder
from numba import prange, jit


encoder = get_encoder()
hparams = json.load(open("hparams.json"))

with open('params.pkl', 'rb') as f:
    params = pickle.load(f)

dev = torch_directml.device(1)
# dev = 'cpu'
dtype = th.float32

wte = th.tensor(params['model/wte']).type(dtype)
wte_gpu = wte.to(dev)
wpe = th.tensor(params['model/wpe']).type(dtype)
ln_f_b = th.tensor(params['model/ln_f/b']).to(dev).type(dtype)
ln_f_g = th.tensor(params['model/ln_f/g']).to(dev).type(dtype)
blocks = []
for i in range(hparams['n_layer']):
    b = {}
    b['attn/c_attn/b'] = th.tensor(params[f'model/h{i}/attn/c_attn/b']).to(dev).type(dtype)
    b['attn/c_attn/w'] = th.tensor(params[f'model/h{i}/attn/c_attn/w']).to(dev).type(dtype)
    b['attn/c_proj/b'] = th.tensor(params[f'model/h{i}/attn/c_proj/b']).to(dev).type(dtype)
    b['attn/c_proj/w'] = th.tensor(params[f'model/h{i}/attn/c_proj/w']).to(dev).type(dtype)

    b['ln_1/b'] = th.tensor(params[f'model/h{i}/ln_1/b']).to(dev).type(dtype)
    b['ln_1/g'] = th.tensor(params[f'model/h{i}/ln_1/g']).to(dev).type(dtype)

    b['ln_2/b'] = th.tensor(params[f'model/h{i}/ln_2/b']).to(dev).type(dtype)
    b['ln_2/g'] = th.tensor(params[f'model/h{i}/ln_2/g']).to(dev).type(dtype)

    b['mlp/c_fc/b'] = th.tensor(params[f'model/h{i}/mlp/c_fc/b']).to(dev).type(dtype)
    b['mlp/c_fc/w'] = th.tensor(params[f'model/h{i}/mlp/c_fc/w']).to(dev).type(dtype)
    b['mlp/c_proj/b'] = th.tensor(params[f'model/h{i}/mlp/c_proj/b']).to(dev).type(dtype)
    b['mlp/c_proj/w'] = th.tensor(params[f'model/h{i}/mlp/c_proj/w']).to(dev).type(dtype)
    blocks.append(b)


times = {}

def get_time(f):
    fname = f.__name__
    if fname not in times:
        times[fname] = 0
    def inner(*arg,**kwarg):
        begin = time.time()
        res = f(*arg,**kwarg)
        duration = time.time() - begin
        times[fname] += duration
        return res
    return inner

@get_time
def softmax(*arg, **kargv):
    th.softmax(*arg, **kargv)

@get_time
def layer_norm(x, g, b, out, eps: float = 1e-5):
    mean = th.mean(x, dim=-1, keepdim=True)
    variance = th.var(x, dim=-1, keepdim=True)
    th.subtract(x, mean, out=out)
    out /= th.sqrt(variance + eps)
    out *= g
    out += b

@get_time
def ffn(x, out, layer):  # [n_seq, n_embd] -> [n_seq, n_embd]
    block = blocks[layer]

    w = block['mlp/c_fc/w']
    b = block['mlp/c_fc/b']
    a = th.empty([x.size()[0], w.size()[1]], dtype=dtype, device=dev)
    th.matmul(x, w, out=a)
    a += b
    th.nn.functional.gelu(a, out=a)

    w = block['mlp/c_proj/w']
    b = block['mlp/c_proj/b']
    r = th.empty([a.size()[0], w.size()[1]], dtype=dtype, device=dev)
    th.matmul(a, w, out=out)
    out += b


@get_time
# @jit
def attention(q, k, v, mask, r, out):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v])]
    # r = th.empty([q.size()[0], k.size()[0]], dtype=dtype, device=dev)
    th.matmul(q, k.T, out=r)
    r /= 8 # th.reciprocal(th.sqrt(th.tensor(q.size()[-1])))
    r += mask
    softmax(r, dim=-1, out = r)
    # th.softmax(r, dim=-1, out = r)
    # r.copy_(out)
    th.matmul(r, v, out=out)
    # th.matmul(r, v, out=out)

def tri(N, dtype):
    return th.tril(th.ones([N, N], dtype=dtype)).to(dev)

list_time = 0

@get_time
def mha(x, mem, layer):  # [n_seq, n_embd] -> [n_seq, n_embd]
    global list_time
    start = time.time()
    block = blocks[layer]
    # qkv projection
    w = block['attn/c_attn/w']
    b = block['attn/c_attn/b']
    th.matmul(x, w, out=mem['pre_qkv'])
    mem['pre_qkv'] += b

    l = x.size()[0]
    qkv_heads = mem['pre_qkv'].reshape(l, 3, hparams['n_head'], -1)
    qkv_heads = th.permute(qkv_heads, [1, 2, 0, 3])
    list_time += time.time() - start

    for i in range(hparams['n_head']):
        q, k, v = qkv_heads[0][i], qkv_heads[1][i], qkv_heads[2][i]
        attention(q, k, v, mem['causal_mask'], mem['attn_mem'], mem['out_head'][:, 64*i:64*(i+1)])

    start = time.time()
    w = block['attn/c_proj/w']
    b = block['attn/c_proj/b']
    th.matmul(mem['out_head'], w, out=mem['mha_out'])
    mem['mha_out'] += b
    list_time += time.time() - start
    # return mem['mha_out']

allot_time = 0

@get_time
def gpt2(inputs):
    global allot_time
    start = time.time()
    x = th.empty([len(inputs), 768], dtype=dtype)
    th.index_select(wte, 0, th.tensor(inputs), out=x)
    x += wpe[th.arange(len(inputs))]
    x = x.to(dev)
    
    layer_norm_mem = th.empty_like(x, device=dev, dtype=dtype)
    mha_mem = {
        'pre_qkv': th.empty([x.size()[0], 2304], device=dev, dtype=dtype),
        'out_head': th.empty([x.size()[0], 768], device=dev, dtype=dtype),
        'attn_mem': th.empty([x.size()[0], x.size()[0]], device=dev, dtype=dtype),
        'mha_out': th.empty([x.size()[0], 768], device=dev, dtype=dtype),
        'causal_mask': (1 - tri(x.shape[0], dtype=x.dtype)) * -1e10
    }
    ffn_mem = th.empty_like(x, device=dev, dtype=dtype)

    allot_time += time.time() - start

    for layer in range(hparams['n_layer']):
        block = blocks[layer]
        b = block['ln_1/b']
        g = block['ln_1/g']
        layer_norm(x, g, b, layer_norm_mem)
        # layer_norm_mem = th.nn.functional.layer_norm(x, g.size(), g, b, eps=1e-5)
        # mha(layer_norm_mem, mha_mem, layer)
        #########################################################
        global list_time
        start = time.time()
        # qkv projection
        w = block['attn/c_attn/w']
        b = block['attn/c_attn/b']
        th.matmul(layer_norm_mem, w, out=mha_mem['pre_qkv'])
        mha_mem['pre_qkv'] += b

        l = x.size()[0]
        qkv_heads = mha_mem['pre_qkv'].reshape(l, 3, hparams['n_head'], -1)
        qkv_heads = th.permute(qkv_heads, [1, 2, 0, 3])
        list_time += time.time() - start

        for i in range(hparams['n_head']):
            q, k, v = qkv_heads[0][i], qkv_heads[1][i], qkv_heads[2][i]
            attention(q, k, v, mha_mem['causal_mask'], mha_mem['attn_mem'], mha_mem['out_head'][:, 64*i:64*(i+1)])

        start = time.time()
        w = block['attn/c_proj/w']
        b = block['attn/c_proj/b']
        th.matmul(mha_mem['out_head'], w, out=mha_mem['mha_out'])
        mha_mem['mha_out'] += b
        list_time += time.time() - start
        #########################################################
        x += mha_mem['mha_out']

        b = block['ln_2/b']
        g = block['ln_2/g']
        layer_norm(x, g, b, layer_norm_mem)
        # layer_norm_mem = th.nn.functional.layer_norm(x, g.size(), g, b, eps=1e-5)
        ffn(layer_norm_mem, ffn_mem, layer)
        # th.add(x, ffn_mem, out=x)
        x += ffn_mem

    b = ln_f_b
    g = ln_f_g
    # layer_norm_mem = th.nn.functional.layer_norm(x, g.size(), g, b, eps=1e-5)
    layer_norm(x, g, b, layer_norm_mem)  # [n_seq, n_embd] -> [n_seq, n_embd]
    r = layer_norm_mem @ wte_gpu.T
    return r


def generate(inputs, n_tokens_to_generate):
    for _ in range(n_tokens_to_generate):
        logits = gpt2(inputs)  # model forward pass
        # logits = logits.cpu()
        next_id = th.argmax(logits[-1]).cpu()  # greedy sampling
        word = encoder.decode([int(next_id)])
        print(word, end='')
        sys.stdout.flush()
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids


def main(prompt: str, n_tokens_to_generate: int = 40):
    input_ids = encoder.encode(prompt)
    output_ids = generate(input_ids, n_tokens_to_generate)
    # output_text = encoder.decode(output_ids)
    # print(output_text)
    print()


if __name__ == '__main__':
    import time
    begin = time.time()
    with th.no_grad():
        main('Alan Turing theorized that computers would one day become')
    duration = time.time() - begin
    print(f'total: {duration:.2f}s')

    for k, v in times.items():
        print(k, v)
    print('allot: ', allot_time)
    print('list_time: ', list_time)
