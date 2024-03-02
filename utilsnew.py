import json
import os
import re

import numpy as np
import torch as th
import requests
import tensorflow as tf
from tqdm import tqdm

from encoder import get_encoder


def load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams):
    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d

    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        print(name)
        # array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        # array = th.tensor(array)
        # name = name[len("model/") :]
        # if name.startswith("h"):
        #     m = re.match(r"h([0-9]+)/(.*)", name)
        #     n = int(m[1])
        #     sub_name = m[2]
        #     set_in_nested_dict(params["blocks"][n], sub_name.split("/"), array)
        # else:
        #     set_in_nested_dict(params, name.split("/"), array)
    exit()
    return params

import pickle

def load_encoder_hparams_and_params():
    model_size = '124M'
    models_dir = 'D:\\code\\picoGPT\\models'

    model_dir = os.path.join(models_dir, model_size)
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)

    encoder = get_encoder(model_size, models_dir)
    hparams = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams)

    # with open('hparams.pkl', 'rb') as f:
    #     hparams = pickle.load(f)

    # with open('params.pkl', 'rb') as f:
    #     params = pickle.load(f)


    return encoder, hparams, params


if __name__ == '__main__':
    encoder, hparams, params = load_encoder_hparams_and_params()
    # import pickle
    # with open('encoder.pkl', 'wb') as f:
    #     pickle.dump(encoder, f)

    # with open('hparams.pkl', 'wb') as f:
    #     pickle.dump(hparams, f)

    # with open('params.pkl', 'wb') as f:
    #     pickle.dump(params, f)
    