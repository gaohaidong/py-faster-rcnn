#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compress a Fast R-CNN network using truncated SVD."""

import _init_paths
import caffe
import argparse
import numpy as np
import os, sys

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Compress a Fast R-CNN network')
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the uncompressed network',
                        default=None, type=str)
    parser.add_argument('--def-svd', dest='prototxt_svd',
                        help='prototxt file defining the SVD compressed network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to compress',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def compress_weights(W, l):
    """Compress the weight matrix W of an inner product (fully connected) layer
    using truncated SVD.

    Parameters:
    W: N x M weights matrix
    l: number of singular values to retain

    Returns:
    Ul, L: matrices such that W \approx Ul*L
    """

    # numpy doesn't seem to have a fast truncated SVD algorithm...
    # this could be faster
    U, s, V = np.linalg.svd(W, full_matrices=False)

    Ul = U[:, :l]
    sl = s[:l]
    Vl = V[:l, :]

    L = np.dot(np.diag(sl), Vl)
    return Ul, L

def compress_conv_layer(net, net_svd, out, layer_name, svd_layer_name_L='', svd_layer_name_U=''):
    if svd_layer_name_L == '':
        svd_layer_name_L = layer_name + '_L'
    if svd_layer_name_U == '':
        svd_layer_name_U = layer_name + '_U'
    if net_svd.params.has_key(svd_layer_name_L):
        l_conv = net_svd.params['svd_layer_name_L'][0].data.shape[0]
        print('  {} bottleneck size: {}'.format(svd_layer_name_L, l_conv))

        # uncompressed weights and biases
        W_conv = net.params[layer_name][0].data
        B_conv = net.params[layer_name][1].data
        W_conv = W_conv.reshape(W_conv.shape[0], -1)

        print('  compressing {}...'.format(layer_name))
        Ul_conv, L_conv = compress_weights(W_conv, l_conv)

        # assert(len(net_svd.params['svd_layer_name_L']) == 1)

        # install compressed matrix factors (and original biases)
        net_svd.params[svd_layer_name_L][0].data[...] = L_conv.reshape(net_svd.params[svd_layer_name_L][0].data.shape)

        net_svd.params[svd_layer_name_U][0].data[...] = Ul_conv.reshape(net_svd.params[svd_layer_name_U][0].data.shape)
        net_svd.params[svd_layer_name_U][1].data[...] = B_conv

        out += '_{}_{}'.format(layer_name, l_conv)
    return out

def compress_fc_layer(net, net_svd, out, layer_name, svd_layer_name_L='', svd_layer_name_U=''):
    if svd_layer_name_L == '':
        svd_layer_name_L = layer_name + '_L'
    if svd_layer_name_U == '':
        svd_layer_name_U = layer_name + '_U'

    if net_svd.params.has_key(svd_layer_name_L):
        l_fc = net_svd.params[svd_layer_name_L][0].data.shape[0]
        print('  {} bottleneck size: {}'.format(svd_layer_name_L, l_fc))

        # uncompressed weights and biases
        W_fc = net.params[layer_name][0].data
        B_fc = net.params[layer_name][1].data

        print('  compressing {}...'.format(layer_name))
        Ul_fc, L_fc = compress_weights(W_fc, l_fc)

        # assert(len(net_svd.params[svd_layer_name_L]) == 1)

        # install compressed matrix factors (and original biases)
        net_svd.params[svd_layer_name_L][0].data[...] = L_fc

        net_svd.params[svd_layer_name_U][0].data[...] = Ul_fc
        net_svd.params[svd_layer_name_U][1].data[...] = B_fc

        out += '_{}_{}'.format(layer_name, l_fc)
    return out

def main():
    args = parse_args()

    # prototxt = 'models/VGG16/test.prototxt'
    # caffemodel = 'snapshots/vgg16_fast_rcnn_iter_40000.caffemodel'
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

    # prototxt_svd = 'models/VGG16/svd/test_fc6_fc7.prototxt'
    # caffemodel = 'snapshots/vgg16_fast_rcnn_iter_40000.caffemodel'
    net_svd = caffe.Net(args.prototxt_svd, args.caffemodel, caffe.TEST)

    print('Uncompressed network {} : {}'.format(args.prototxt, args.caffemodel))
    print('Compressed network prototxt {}'.format(args.prototxt_svd))

    out = os.path.splitext(os.path.basename(args.caffemodel))[0] + '_svd'
    out_dir = os.path.dirname(args.caffemodel)

    # Compress conv layers
    conv_names = ['conv{}_{}'.format(i, j) for i in [4, 5] for j in [1, 2, 3]]
    for conv_name in conv_names:
        out = compress_conv_layer(net, net_svd, out, conv_name)


    # Compress fc layers
    fc_names = ['fc{}'.format(i) for i in [6, 7]]
    for fc_name in fc_names:
        out = compress_fc_layer(net, net_svd, out, fc_name)

    filename = '{}/{}.caffemodel'.format(out_dir, out)
    net_svd.save(filename)
    print 'Wrote svd model to: {:s}'.format(filename)

if __name__ == '__main__':
    main()
