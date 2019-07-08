import re
import os
from itertools import combinations, permutations

import numpy as np
# import tensorflow as tf
# from pymystem3 import Mystem
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences



def ctx_permutations(ctx, ws, pad_with='#'):
    le, ri = ws
    ctxs = []
    cache = set()

    for i, x in enumerate(ctx):
        if x == pad_with or x in cache:
            continue

        cache.add(x)
        plchld = [[], [], []]
        plchld[1].append(x)
        
        for j, y in enumerate(ctx):
            if i == j:
                continue

            if len(plchld[0]) < le:
                plchld[0].append(y)

            else:
                plchld[2].append(y)

        ctxs.append(plchld)
    
    return ctxs 


def ctx_iter(text, ws, step=1, cntr_sz=1, padding=True, pad_with='#'):
    le, ri = ws
    hi = len(text)

    if isinstance(text, str):
        text = list(text)

    if padding:
        text = [pad_with] * le + text + [pad_with] * ri

    i = 0
    k = le
    j = le + cntr_sz

    while i < hi:
        ctx = []
        ctx.extend(text[i:k])
        ctx.extend(text[k:j])
        ctx.extend(text[j:j+ri])
        ctxs = ctx_permutations(ctx, ws)
        
        yield from ctxs

        i += step
        k += step
        j += step


data = [
    'раз два три четрые',
    'пять шесть',
    'семь восемь девять',
    'десять одиннадцать двенадцать',
]


def ctx_update(ctx, i2ctx, t2i):
    new_ctx = []

    for c_ctx in ctx:
        c_new_ctx = []

        for t in c_ctx:
            if t not in t2i:
                t2i[t] = len(t2i)

            c_new_ctx.append(t2i[t])

        new_ctx.append(c_new_ctx)


    i2ctx[len(i2ctx)] = new_ctx 


def create_maps(data, ws):
    i2ctx = {}
    t2i = {}
    step = sum(ws) + 1

    for d in data:
        for ctx in ctx_iter(d.split(), ws=ws, step=step):
            ctx_update(ctx, i2ctx, t2i) 

    return i2ctx, t2i


def batch_iter(i2ctx, t2i, batch_size):
    ctx_idx_cache = set()
    ctxs = []
    hi = len(i2ctx)
    
    c_batch_flag = False
    c_batch_ctxs = []

    while len(ctx_idx_cache) < len(i2ctx):
        ctx_idx = np.random.randint(0, hi)

        while ctx_idx in ctx_idx_cache:
            ctx_idx = np.random.randint(0, hi)

        ctx_idx_cache.add(ctx_idx)
        c_batch_ctxs.append(ctx_idx)

        if len(c_batch_ctxs) == batch_size:
            ctxs.append(c_batch_ctxs)
            c_batch_ctxs = []

    # residual batch of arbitrary size when `len(i2ctx) % batch_size != 0` 
    if c_batch_ctxs:
        ctxs.append(c_batch_ctxs)

    return ctxs

# for x in ctx_iter('abcdefgh', (1, 1), step=3): print(x, '\n')
i2ctx, t2i = create_maps(data, (1, 1))
ctxs = batch_iter(i2ctx, t2i, batch_size=2)
print(list(ctxs))
print(i2ctx[0])