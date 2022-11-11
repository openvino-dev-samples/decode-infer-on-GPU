# decode_infer-on-GPU
This sample shows how to use the oneAPI Video Processing Library (oneVPL) to perform a ***single and mulit-source*** video decode and preprocess and inference using OpenVINO to show the device surface sharing (zero copy), modified from the example in (oneVPL)[https://github.com/oneapi-src/oneVPL/tree/master/examples].

![onvpl_openvino](https://user-images.githubusercontent.com/91237924/200984002-1b469ec9-b722-4f83-b357-ef1c40c5a439.png)

## System requirements

| Optimized for    | Description
|----------------- | ----------------------------------------
| OS               | Ubuntu* 20.04
| Hardware         | Compatible with Intel® oneAPI Video Processing Library(oneVPL) GPU implementation, which can be found at https://github.com/oneapi-src/oneVPL-intel-gpu 
| Software         | Intel® oneAPI Video Processing Library(oneVPL) CPU implementation and Intel® - OpenVINO 2022.2

## How to build the sample

### Install OpenVINO toolkits 2022.2 from achieved package
1) Download and install OpenVINO C++ runtime:
https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_from_archive_linux.html
2) Configurations for Intel® Processor Graphics (GPU):
https://docs.openvino.ai/latest/openvino_docs_install_guides_configurations_for_intel_gpu.html#gpu-guide
3) Install the OpenVINO Developtment tools:
```shell
pip install openvino-dev[tensorflow2,onnx]
```

### Install environment package prerequisites (optional)
```shell
apt update && \
    apt install --no-install-recommends -q -y software-properties-common gnupg wget sudo unzip libnss3-tools ncurses-term python3-pip
```

### Install Intel graphics stack packages from Agama
```shell
wget https://repositories.intel.com/graphics/intel-graphics.key && \
    apt-key add intel-graphics.key && \
    apt-add-repository 'deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main' && \
    apt update && \
    apt install -y libmfx1 libmfxgen1 vainfo intel-opencl-icd intel-level-zero-gpu level-zero intel-media-va-driver-non-free python3.9
```

### Install Dev package
```shell
apt install -y cmake build-essential libva-dev libdrm-dev net-tools pkg-config libigc-dev intel-igc-cm libigdfcl-dev libigfxcmrt-dev level-zero-dev opencl-headers build-essential
```

Or you can follow this instruction to install the driver package for a discrete GPU device.
https://dgpu-docs.intel.com/index.html

### Install oneVPL devkit package from oneAPI
```shell
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    echo "deb https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt update && apt install -y intel-oneapi-onevpl-devel  
```

### Configure the Environment
1) OpenVINO:
```shell
source <INSTALL_DIR>/setupvars.sh
```
2) oneAPI:
```shell
source /opt/intel/oneapi/setvars.sh
```

3） Build the source code
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

### Download test model
Download the vehicle detection model from OpenVINO model zoo
```
omz_downloader --name vehicle-detection-0200
```

### Run the program
- For single source 
```
./single_src/single_source -i ../content/cars_320x240.h265 -m ~/vehicle-detection-0200/FP32/vehicle-detection-0200.xml 
```
- For multiple source 
```
./multi_src/multi_source -i ../content/cars_320x240.h265,../content/cars_320x240.h265,../content/cars_320x240.h265 -m ~/vehicle-detection-0200/FP32/vehicle-detection-0200.xml -bs 2 -nr 4
```

- -i = Path to one or multiple input video files, separated by comma;
- -m = Path to IR .xml file;
- -bs = Batch size;
- -nr = Number inference requests;
- -fr = Number of frame to be decoded for each input source;

Tips: Since the sample has been set the number of stream as 1, the number of infer request should be larger than 1.

## Example of Output
In this sample, you will get the inference result according to stream id of input source.
```
libva info: VA-API version 1.12.0
libva info: Trying to open /opt/intel/mediasdk/lib64/iHD_drv_video.so
libva info: Found init function __vaDriverInit_1_12
libva info: va_openDriver() returns 0
Implementation details:
  ApiVersion:           2.7  
  Implementation type:  HW
  AccelerationMode via: VAAPI
  Path: /usr/lib/x86_64-linux-gnu/libmfx-gen.so.1.2.7

Frames [stream_id=1] [stream_id=0]
image0: bbox 204.99, 49.43, 296.43, 144.56, confidence = 0.99805
image0: bbox 91.26, 115.56, 198.41, 221.69, confidence = 0.99609
image0: bbox 36.50, 44.75, 111.34, 134.57, confidence = 0.98535
image0: bbox 77.92, 72.38, 155.06, 164.30, confidence = 0.97510
image1: bbox 204.99, 49.43, 296.43, 144.56, confidence = 0.99805
image1: bbox 91.26, 115.56, 198.41, 221.69, confidence = 0.99609
image1: bbox 36.50, 44.75, 111.34, 134.57, confidence = 0.98535
image1: bbox 77.92, 72.38, 155.06, 164.30, confidence = 0.97510
Frames [stream_id=1] [stream_id=0]
image0: bbox 206.96, 50.41, 299.54, 146.23, confidence = 0.99805
image0: bbox 93.81, 115.29, 200.86, 222.94, confidence = 0.99414
image0: bbox 84.15, 92.91, 178.14, 191.82, confidence = 0.99316
image0: bbox 37.78, 45.82, 113.29, 132.28, confidence = 0.98193
image0: bbox 75.96, 71.88, 154.31, 164.54, confidence = 0.96582
image1: bbox 206.96, 50.41, 299.54, 146.23, confidence = 0.99805
image1: bbox 93.81, 115.29, 200.86, 222.94, confidence = 0.99414
image1: bbox 84.15, 92.91, 178.14, 191.82, confidence = 0.99316
image1: bbox 37.78, 45.82, 113.29, 132.28, confidence = 0.98193
image1: bbox 75.96, 71.88, 154.31, 164.54, confidence = 0.96582
...
decoded and infered 60 frames
Time = 0.328556s
```
