import tensorflow as tf
from casadi import *
import numpy as np

def eval_nn(model, layer_out):
    config = model.get_config()

    # For each layer in the layers list:
    for layer_k in config['layers']:
        # If the output of a given layer is already in the inputs, skip this layer.
        if layer_k['name'] in layer_out.keys():
            pass

        # Type: InputLayer
        elif layer_k['class_name'] == 'InputLayer':
            pass

        # Type: Dense layer
        elif layer_k['class_name'] == 'Dense':
            weights = model.get_layer(layer_k['name']).get_weights()

            name_in = layer_k['inbound_nodes'][0][0][0]
            tensor_in = layer_out[name_in]

            tensor_out = tensor_in@weights[0]
            if layer_k['config']['use_bias']:
                tensor_out = tensor_out + weights[1].reshape(1,-1)

            if layer_k['config']['activation'] == 'tanh':
                tensor_out = np.tanh(tensor_out)
            elif layer_k['config']['activation'] == 'relu':
                tensor_out = np.maximum(tensor_out,0)
            elif layer_k['config']['activation'] == 'linear':
                pass
            else:
                raise Exception('Unknown activation.')


            layer_out[layer_k['name']] = tensor_out

        # Type: Concatenate layer
        elif layer_k['class_name'] == 'Concatenate':
            concatenate_list = []
            sym_type_list = []
            for input_j in layer_k['inbound_nodes'][0]:
                tensor_in = layer_out[input_j[0]]
                concatenate_list.append(tensor_in)
                sym_type_list.append(isinstance(tensor_in, (casadi.MX, casadi.SX, casadi.DM)))

            if any(sym_type_list):
                tensor_out = horzcat(*concatenate_list)
            else:
                tensor_out = np.concatenate(concatenate_list, axis=1)
            layer_out[layer_k['name']] = tensor_out

        # Type: Slicing layer
        elif layer_k['class_name'] == 'TensorFlowOpLayer':
            # Get slicing indices (stored in constants)
            start = layer_k['config']['constants'][1]
            stop = layer_k['config']['constants'][2]
            step = layer_k['config']['constants'][3]

            # Prepare slicing objects and store them in a list:
            slice_list = []
            for start_i, step_i, stop_i in zip(start,step,stop):
                # Slicing is coded: 0 means None
                if start_i == 0: start_i = None
                if step_i == 0: step_i = None
                if stop_i == 0: stop_i = None
                slice_list.append(slice(start_i, stop_i, step_i))

            slice_tuple = tuple(slice_list)

            # Get and slice inbound node:
            name_in = layer_k['inbound_nodes'][0][0][0]
            tensor_in = layer_out[name_in]
            tensor_out = tensor_in[slice_tuple]

            layer_out[layer_k['name']] = tensor_out

        elif layer_k['class_name'] =='Add':
            # Add layer:
            tensor_out = 0
            for name_in in layer_k['inbound_nodes'][0]:
                tensor_in = layer_out[name_in[0]]
                tensor_out += tensor_in

            layer_out[layer_k['name']] = tensor_out

        else:
            raise Exception('Unsupported layer of type {}'.format(layer_k['class_name']))

    return layer_out
