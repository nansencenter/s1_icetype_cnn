# s1_icetype_cnn
Retrieve sea ice type from Sentinel-1 SAR with CNN

# Usage
1. Perform thermal/texture noise and angular correction

`s1_correction.py S1B_EW_GRDM_1SDH_20200323T053203_20200323T053303_020815_027780_8729.zip S1B_EW_GRDM_1SDH_20200323T053203.tif`

2. Apply CNN

`./get_icetype.py weights.hdf5 S1B_EW_GRDM_1SDH_20200323T053203.tif S1B_EW_GRDM_1SDH_20200323T053203.npy`
