# Model exporter to C binary format
import os
import struct
import argparse
import json

import numpy as np
import torch



def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)


def serialize_int8(file, tensor):
    """ writes one int8 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)

def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float() # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:,None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:,None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr


def write_weights(file, model, key):
    """ writes the layer weights to file """
    print(f"writing {key} {list(model[key].shape)[::-1]}")
    serialize_fp32(file, model[key])

def write_weights_q8_0(file, model, key, group_size=64):
    """ writes the quantized layer weights to file """
    q, s, err = quantize_q80(model[key], group_size)

    serialize_int8(file, q)
    serialize_fp32(file, s)

    print(f"{key} quantized {tuple(model[key].shape)} to Q8_0 with max error {err}")


def write_layer_weights(file, model, layer, n_layers):
    """ writes the layer weights to file """
    for n in range(n_layers):
        write_weights(file, model, layer % n)


def write_layer_weights_q8_0(file, model, layer, n_layers, group_size=64):
    #""" writes the layer weights to file """
    #for n in range(n_layers):
    #    write_weights_q8_0(file, model, layer % n, group_size)

    qtensors = { "q": [], "s": [] }

    for n in range(n_layers):
        q, s, err = quantize_q80(model[layer % n], group_size)

        qtensors["q"].append(q)
        qtensors["s"].append(s)

        print(f"{layer % n} quantized {tuple(model[layer % n].shape)} to Q8_0 with max error {err}")

    for q in qtensors["q"]:
        serialize_int8(file, q)

    for s in qtensors["s"]:
        serialize_fp32(file, s)



def load_config(config_path):
    with open(config_path) as f:
        config = json.load(f)

    return config

def load_model(model_path):
    model = torch.load(model_path, map_location='cpu')

     # remove the 'backbone.' prefix from the keys
    unwanted_prefix = 'backbone.'
    for k,v in list(model.items()):
        if k.startswith(unwanted_prefix):
            model[k[len(unwanted_prefix):]] = model.pop(k)

    return model



def export_model(model, config, output_path):
    out_file = open(output_path, 'wb')

    n_layers = config['n_layer']

    '''
    Example of the model structure:
    embedding.weight - [50280, 768]
    layers.0.mixer.D - [1536]
    layers.0.mixer.in_proj.weight - [3072, 768]
    layers.0.mixer.conv1d.weight - [1536, 1, 4]
    layers.0.mixer.conv1d.bias - [1536]
    layers.0.mixer.x_proj.weight - [80, 1536]
    layers.0.mixer.dt_proj.weight - [1536, 48]
    layers.0.mixer.dt_proj.bias - [1536]
    layers.0.mixer.A_log - [1536, 16]
    layers.0.mixer.out_proj.weight - [768, 1536]
    layers.0.norm.weight - [768]
    norm_f.weight - [768]
    lm_head.weight - [50280, 768]
    '''

	for n in range(n_layers):
    	a_log = f'layers.{n}.mixer.A_log'
    	if a_log in model:
        	model[f'layers.{n}.mixer.A'] = -torch.exp(model.pop(a_log))


    write_weights(out_file, model, 'embedding.weight')

    write_layer_weights(out_file, model, 'layers.%d.mixer.in_proj.weight', n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.conv1d.weight', n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.conv1d.bias', n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.x_proj.weight', n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.dt_proj.weight', n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.dt_proj.bias', n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.A', n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.D', n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.out_proj.weight', n_layers)
    write_layer_weights(out_file, model, 'layers.%d.norm.weight', n_layers)

    write_weights(out_file, model, 'norm_f.weight')
    write_weights(out_file, model, 'lm_head.weight')

    out_file.close()
    print(f"Exported model to {output_path}")


def export_model_q8_0(model, config, output_path, group_size=64):
    out_file = open(output_path, 'wb')

    n_layers = config['n_layer']

    '''
    Example of the model structure:
    embedding.weight - [50280, 768]
    layers.0.mixer.D - [1536]
    layers.0.mixer.in_proj.weight - [3072, 768]
    layers.0.mixer.conv1d.weight - [1536, 1, 4]
    layers.0.mixer.conv1d.bias - [1536]
    layers.0.mixer.x_proj.weight - [80, 1536]
    layers.0.mixer.dt_proj.weight - [1536, 48]
    layers.0.mixer.dt_proj.bias - [1536]
    layers.0.mixer.A_log - [1536, 16]
    layers.0.mixer.out_proj.weight - [768, 1536]
    layers.0.norm.weight - [768]
    norm_f.weight - [768]
    lm_head.weight - [50280, 768]
    '''

	for n in range(n_layers):
    	a_log = f'layers.{n}.mixer.A_log'
    	if a_log in model:
        	model[f'layers.{n}.mixer.A'] = -torch.exp(model.pop(a_log))


    write_weights_q8_0(out_file, model, 'embedding.weight')

    write_layer_weights_q8_0(out_file, model, 'layers.%d.mixer.in_proj.weight', n_layers)
    
    write_layer_weights(out_file, model, 'layers.%d.mixer.conv1d.weight', n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.conv1d.bias', n_layers)

    write_layer_weights_q8_0(out_file, model, 'layers.%d.mixer.x_proj.weight', n_layers)

    write_layer_weights(out_file, model, 'layers.%d.mixer.dt_proj.weight', n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.dt_proj.bias', n_layers)

    write_layer_weights(out_file, model, 'layers.%d.mixer.A', n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.D', n_layers)

    write_layer_weights_q8_0(out_file, model, 'layers.%d.mixer.out_proj.weight', n_layers)

    write_layer_weights(out_file, model, 'layers.%d.norm.weight', n_layers)
    write_weights(out_file, model, 'norm_f.weight')

    write_weights_q8_0(out_file, model, 'lm_head.weight')

    out_file.close()
    print(f"Exported model to {output_path}")





def export_config(model, config, output_path):
    """
    Exports the config to a C header file, following this configuration example:

        #define VOCAB_SIZE 256
        #define N_LAYER 12
        #define D_MODEL 768
        #define D_INNER 1536
        #define DT_RANK 48
        #define D_STATE 16
        #define D_CONV 4

    #define [KEY] [VALUE]
    key is converted to uppercase and value is the value from the config dictionary
    """

    vocab_size = config['vocab_size']
    rounded_vocab_size = vocab_size if vocab_size % 8 == 0 else vocab_size + (8 - (vocab_size % 8))

    with open(output_path, 'w') as f:
        f.write("#pragma once\n\n")
        f.write("#define VOCAB_SIZE %d\n" % vocab_size)
        f.write("#define ROUNDED_VOCAB_SIZE %d\n\n" % rounded_vocab_size)
        f.write("#define N_LAYER %d\n" % config['n_layer'])
        f.write("#define D_MODEL %d\n" % config['d_model'])
        f.write("#define D_INNER %d\n" % (2 * config['d_model']))
        f.write("#define DT_RANK %d\n" % model['layers.0.mixer.dt_proj.weight'].shape[1])
        f.write("#define D_STATE %d\n" % model['layers.0.mixer.A'].shape[1])
        f.write("#define D_CONV %d\n" % model['layers.0.mixer.conv1d.weight'].shape[2])

    print(f"Exported config to {output_path}")
