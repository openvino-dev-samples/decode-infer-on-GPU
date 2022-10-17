# decode_infer-on-GPU
This sample shows how to use the oneAPI Video Processing Library (oneVPL) to perform a simple video decode and preprocess and inference using OpenVINO to show the device surface sharing (zero copy), modified from the example in (oneVPL)[https://github.com/oneapi-src/oneVPL/tree/master/examples].

![onevpl](https://user-images.githubusercontent.com/91237924/195313571-cfd0fa36-74d5-4097-8c92-a78134462a22.png)

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
```shell

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

- Build the source code
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

### Download test model
Download the vehicle detection model from OpenVINO model zoo
```
omz_downloader -m vehicle-detection-0200
```

### Run the program
```
./infer-decode -i ../content/cars_320x240.h265 -m ~/vehicle-detection-0200/FP32/vehicle-detection-0200.xml 
```

## Example of Output
```
libva info: VA-API version 1.12.0
libva info: Trying to open /opt/intel/mediasdk/lib64/iHD_drv_video.so
libva info: Found init function __vaDriverInit_1_12
libva info: va_openDriver() returns 0
libva info: VA-API version 1.12.0
libva info: Trying to open /opt/intel/mediasdk/lib64/iHD_drv_video.so
libva info: Found init function __vaDriverInit_1_12
libva info: va_openDriver() returns 0
Implementation details:
  ApiVersion:           2.7  
  Implementation type:  HW
  AccelerationMode via: VAAPI
  Path: /usr/lib/x86_64-linux-gnu/libmfx-gen.so.1.2.7

libva info: VA-API version 1.12.0
libva info: Trying to open /opt/intel/mediasdk/lib64/iHD_drv_video.so
libva info: Found init function __vaDriverInit_1_12
libva info: va_openDriver() returns 0
Decoding VPP, and infering ../content/cars_320x240.h265 with /home/ethan/oneVPL/examples/interop/advanced-decvpp-infer/intel/vehicle-detection-0200/FP16/vehicle-detection-0200.xml
[0,0] element, prob = 0.998047    (205,49)-(296,144) batch id : 0 WILL BE PRINTED!
[1,0] element, prob = 0.996094    (91,115)-(198,221) batch id : 0 WILL BE PRINTED!
[2,0] element, prob = 0.985352    (36,44)-(111,134) batch id : 0 WILL BE PRINTED!
[3,0] element, prob = 0.975098    (77,72)-(154,164) batch id : 0 WILL BE PRINTED!
[4,0] element, prob = 0.463135    (87,99)-(178,178) batch id : 0
...
```
