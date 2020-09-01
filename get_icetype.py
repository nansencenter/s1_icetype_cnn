#!/usr/bin/env python
import argparse
import json
import os
import sys

import gdal
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

def recall(x, y):
    return 0

def parse_args(args):
    """ Parse input arguments and check validity """
    parser = argparse.ArgumentParser(
        description='Retrieve sea ice type from Sentinel-1 SAR data')
    parser.add_argument('wfile', type=str, help='Input filename with CNN weights.')
    parser.add_argument('ifile', type=str, help='Input denoised GeoTIFF filename.')
    parser.add_argument('ofile', type=str, help=(
        'Output filename in GeoTiff or Numpy format depending on the file extension.'))
    parser.add_argument('-s', '--step', type=int, default=50, help=(
        'Processing step. Subimages will be taken from input SAR images at each '
        '<s> step.'))

    args = parser.parse_args(args)

    if not os.path.exists(args.wfile):
        raise ValueError('File %s with CNN weights is not found. Provide correct path and filename.' % args.wfile)
    if not os.path.exists(args.ifile):
        raise ValueError('File %s with corrected S1 data is not found. Provide correct path and filename.' % args.ifile)

    return args

def get_band_number(ds, pol):
    """ Get band number in input dataset <ds> for a given polarisation <pol> """
    band = None
    for i in range(ds.RasterCount):
        metadata = ds.GetRasterBand(i+1).GetMetadata()
        for j in metadata:
            if 'sigma0_%s'%pol in metadata[j]:
                return i+1

def load_file(filename):
    """ Read images with denoised sigma0 in HH and HV polarisations from <filename>

    Returns
    -------
    data : numpy.ndarray [2 x W x H]
        2 sigma0 images in HH and HV polarisations
    ds : gdal.Dataset
        dataset with original Sentinel-1 file

    """
    pols = ['HH', 'HV']
    bands = {}
    ds = gdal.Open(filename)
    bands = {pol:get_band_number(ds, pol) for pol in pols}
    data = [ds.GetRasterBand(bands[pol]).ReadAsArray() for pol in pols]
    return data, ds

def apply_cnn(cnn_filename, data, step):
    """
    Load CNN from hdf5 file, apply to input sigma0 data and generate map with
    ice type probabilities.

    Parameters
    ----------
    cnn_filename : str
        filename of CNN weights
    data : numpy.ndarray [2 x W x H]
        2 sigma0 imaages in HH and HV polarisations
        range of cols in the original sigma0 image
    step : int
        Step to read HH/HV sub-images

    Returns
    -------
    ice_pro : numpy.ndarray [R x C x L]
        Ice type probabilities. R - number of rows, C - number of columns,
        L - number of classes.

    """
    model = load_model(cnn_filename, custom_objects={'recall':recall})
    for layer in model.layers:
        cfg = layer.get_config()
        if 'batch_input_shape' in cfg:
            inp_size = cfg['batch_input_shape'][1]
            break
    out_size = model.layers[-1].get_config()['units']

    rows = range(0, data[0].shape[0]-inp_size, step)
    cols = range(0, data[0].shape[1]-inp_size, step)
    ice_pro = np.zeros((len(rows), len(cols), out_size)) + np.nan
    for i, r in enumerate(rows):
        inp = []
        for c in cols:
            inp.append(np.stack([d[r:r+inp_size, c:c+inp_size] for d in data], 2))
        inp = np.array(inp)
        s0m = inp[:,:,:,0].mean(axis=(1,2))
        gpi = np.where(np.isfinite(s0m))[0]
        if gpi.size > 0:
            ice_pro[i, gpi] = model.predict(inp[gpi])
    return ice_pro

def create_ice_chart(ice_pro, cnn_filename):
    """ Create map with ice types and map with probabilities for probability of each class

    Parameters
    ----------
    ice_pro : 3D numpy.ndarray
        probabilities of each class
    cnn_filename : str
        filename with the CNN weights

    Returns
    -------
    ice_map : 2D numpy.ndarray
        raster with labels of ice types
    pro_map : 2D numpy array
        raster with probability of ice type
    metadata : str
        Explanation of ice type values

    """
    json_filename = cnn_filename.replace('.hdf5', '.json')
    if os.path.exists(json_filename):
        with open(json_filename, 'rt') as f:
            metadata = json.loads(f.read())
    else:
        metadata = {'description': 'unknown'}
    return np.argmax(ice_pro, axis=2), np.max(ice_pro, axis=2), metadata

def prepare_gcps(ds, step):
    """ Prepare GCPs for the output dataset

    Parameters
    ----------
    ds : gdal.Dataset
        dataset with original Sentinel-1 file
    step : int
        Step to read HH/HV sub-images

    Returns
    -------
    gcps : tuple with gdal.GCP
        Destination dataset GDCPs
    """
    gcps = ds.GetGCPs()
    for gcp in gcps:
        gcp.GCPPixel /= step
        gcp.GCPLine /= step
    return gcps

def export_geotiff(dst_filename, ds, ice_map, pro_map, step, metadata, options=('COMPRESS=LZW',)):
    """ Export ice chart into GeoTiff file

    Parameters
    ----------
    dst_filename : str
        Output GeoTiff filename
    ds : gdal.Dataset
        dataset with original Sentinel-1 file
    ice_map : 2D numpy.ndarray
        raster with labels of ice types
    pro_map : 2D numpy array
        raster with probability of ice type
    step : int
        Step to read HH/HV sub-images
    metadata : dict
        Metadata for the ice_type band
    options: tuple of strings
        Options for GDAL GTiff driver

    """
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(dst_filename,
        xsize=ice_map.shape[1],
        ysize=ice_map.shape[0],
        bands=2,
        eType=gdal.GDT_Byte,
        options=list(options))
    gcps = prepare_gcps(ds, step)
    dst_ds.SetGCPs(gcps, ds.GetGCPProjection())
    dst_ds.GetRasterBand(1).WriteArray(ice_map)
    dst_ds.GetRasterBand(1).SetMetadata(metadata)
    dst_ds.GetRasterBand(1).SetMetadataItem('short_name', 'ice_type')
    dst_ds.GetRasterBand(2).WriteArray((pro_map*100).astype('int8'))
    dst_ds.GetRasterBand(2).SetMetadataItem('short_name', 'ice_type_probability')
    dst_ds = None

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    data, ds = load_file(args.ifile)
    ice_pro = apply_cnn(args.wfile, data, args.step)
    ice_map, pro_map, meta = create_ice_chart(ice_pro, args.wfile)
    if 'tif' in os.path.splitext(args.ofile)[1]:
        export_geotiff(args.ofile, ds, ice_map, pro_map, args.step, meta)
    else:
        np.save(args.ofile, ice_pro=ice_pro, ice_map=ice_map, pro_map=pro_map)
