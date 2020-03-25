FROM s1denoise
LABEL maintainer="Anton Korosov <anton.korosov@nersc.no>"
LABEL purpose="Python libs for ice type retrieval from Sentinel-1 TOPSAR using CNN"

RUN conda install -y tensorflow=2.1.0

COPY get_icetype.py /usr/local/bin/

