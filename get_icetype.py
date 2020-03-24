#!/usr/bin/env python
import argparse
import sys

import gdal
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

def recall(x, y):
    return 0

def parse_args(args):
    """ Parse input arguments """
    parser = argparse.ArgumentParser(
        description="Correct Sentinel-1 TOPSAR EW GRDM for thermal and texture noise and angular dependence")
    parser.add_argument('wfile', type=str, help='Input file with CNN model')
    parser.add_argument('ifile', type=str, help='Input denoised GeoTIFF file')
    parser.add_argument('ofile', type=str, help='Output numpy file')
    parser.add_argument('-s', '--step', type=int, default=50, help='Step of processing')
    return parser.parse_args(args)

def get_band_number(ds, pol):
    """ Get band number in input dataset <ds> for a given polarisation <pol> """
    band = None
    for i in range(ds.RasterCount):
        metadata = ds.GetRasterBand(i+1).GetMetadata()
        for j in metadata:
            if 'sigma0_%s'%pol in metadata[j]:
                return i+1

def load_file(filename, size, step):
    """ Read subimages with denoised sigma0 in HH and HV polarisations

    Parameters
    ----------
    filename : str
        input filename
    size : int
        Size of subaimges
    step : int
        Step to read HH/HV sub-images

    Returns
    -------
    data : numpy.ndarray [K x size x size 2]
        Data for CNN

    """
    pols = ['HH', 'HV']
    bands = {}
    ds = gdal.Open(filename)
    bands = {pol:get_band_number(ds, pol) for pol in pols}
    data = {pol: ds.GetRasterBand(bands[pol]).ReadAsArray() for pol in pols}
    data_shape = data[pols[0]].shape
    rows = range(0, data_shape[0]-size, step)
    cols = range(0, data_shape[1]-size, step)
    output = []
    for r in rows:
        for c in cols:
            output.append(np.stack([data[pol][r:r+size, c:c+size] for pol in pols], 2))
    return np.array(output), rows, cols

def apply_cnn(data, rows, cols):
    """ Apply CNN to input data and generate map with ice type probabilities

    Parameters
    ----------
    data : numpy.ndarray [K x size x size 2]
        sigma0 in subimages in HH/HV polarisations. K - number of subimages.
    rows : range
        range of rows in the original sigma0 image
    cols : range
        range of cols in the original sigma0 image

    Returns
    -------
    ice_pro : numpy.ndarray [R x C x L]
        Ice type probabilities. R - number of rows, C - number of columns,
        L - number of classes.

    """
    s0m = data[:,:,:,0].mean(axis=(1,2))
    gpi = np.isfinite(s0m)
    # TODO: add batch processing
    ice_pro_gpi = model.predict(data[gpi])
    ice_pro = np.zeros((data.shape[0], ice_pro_gpi.shape[1])) + np.nan
    ice_pro[gpi] = ice_pro_gpi
    ice_pro.shape = (len(rows), len(cols), ice_pro.shape[1])
    return ice_pro

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    model = load_model(args.wfile, custom_objects={'recall':recall})
    size = model.layers[1].get_config()['batch_input_shape'][1]
    data, rows, cols = load_file(args.ifile, size, args.step)
    ice_pro = apply_cnn(data, rows, cols)
    np.save(args.ofile, ice_pro)
