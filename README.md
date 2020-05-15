# s1_icetype_cnn-1.1
Retrieve sea ice types from corrected Sentinel-1 SAR with convolutional neural networks.

### Run the example Jupyter Notebook

1. Start the official Tensorflow Docker container:
```
docker run -v $PWD:/tf --rm -it -p 8888:8888  tensorflow/tensorflow:latest-jupyter
```
2. Open the provided URL in a browser and open the notebook "cnn_for_sea_ice_types.ipynb"
3. Execute the cells

### Installation and usage on any platform using Docker

1. Build image **s1denoised** as explained here: https://github.com/nansencenter/sentinel1denoised/

2. Build image **s1icetype** using Dockerfile from this repository: `docker build . -t s1icetype`

3. Run thermal noise, texture noise and angular corrections

```
docker run --rm -it -v /path/to/s1/files:/data s1_correction.py /data/S1B_EW_GRDM_1SDH_20200323T053203_20200323T053303_020815_027780_8729.zip /data/S1B_EW_GRDM_1SDH_20200323T053203.tif
```

4. Run the ice type algorithm (mounted directory names can be anything)

```
docker run --rm -it \
    -v /path/to/cnn/file:/cnn \
    -v /path/to/s1/files:/data \
    get_icetype.py /cnn/cnn_weights.hdf5 /data/S1B_EW_GRDM_1SDH_20200323T053203.tif /data/S1B_EW_GRDM_1SDH_20200323T053203_icetype.tif
```

### Installation and usage on Linux using conda

1. Create environment **s1denoise** for S1 correction as explained here: https://github.com/nansencenter/sentinel1denoised/

2. Activate environment **s1denoise**  and install tensorflow 2.1.0

```
conda activate s1denoise
conda install tensorflow=2.1.0
```
3. Run thermal noise, texture noise and angular corrections

```
s1_correction.py S1B_EW_GRDM_1SDH_20200323T053203_20200323T053303_020815_027780_8729.zip S1B_EW_GRDM_1SDH_20200323T053203.tif
```

4. Run ice type algorithm

```
./get_icetype.py cnn_weights.hdf5 S1B_EW_GRDM_1SDH_20200323T053203.tif S1B_EW_GRDM_1SDH_20200323T053203_icetype.tif
```
