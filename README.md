This is forked from https://github.com/jiyanggao/Video-Person-ReID

# Video-Person-ReID

This is the code repository for tech report "Revisiting Temporal Modeling for Video-based Person ReID": https://arxiv.org/abs/1805.02104.
If you find this help your research, please cite

    @article{gao2018revisiting,
      title={Revisiting Temporal Modeling for Video-based Person ReID},
      author={Gao, Jiyang and Nevatia, Ram},
      journal={arXiv preprint arXiv:1805.02104},
      year={2018}
    }

### Introduction
This repository contains PyTorch implementations of temporal modeling methods for video-based person reID. It is forked from [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid).. Based on that, I implement (1) video sampling strategy for training and testing, (2) temporal modeling methods including temporal pooling, temporal attention, RNN and 3D conv. The base loss function and basic training framework remain the same as [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid).

### Motivation
Although previous work proposed many temporal modeling methods and did extensive experiments, but it's still hard for us to have an "apple-to-apple" comparison across these methods. As the image-level feature extractor and loss function are not the same, which have large impact on the final performance. Thus, we want to test the representative methods under an uniform framework.

### Dataset
All experiments are done on MARS, as it is the largest dataset available to date for video-based person reID. Please follow [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) to prepare the data. The instructions are copied here: 

1. Create a directory named `mars/` under `data/`.
2. Download dataset to `data/mars/` from http://www.liangzheng.com.cn/Project/project_mars.html.
3. Extract `bbox_train.zip` and `bbox_test.zip`.
4. Download split information from https://github.com/liangzheng06/MARS-evaluation/tree/master/info and put `info/` in `data/mars` (we want to follow the standard split in [8]). The data structure would look like:
```
mars/
    bbox_test/
    bbox_train/
    info/
```
5. Use `-d mars` when running the training code.

### Usage
To train the model, please run

    python main_video_person_reid.py --arch=resnet50tp
arch could be resnet50tp (Temporal Pooling), resnet50ta (Temporal Attention), resnet50rnn (RNN), resnet503d (3D conv). For 3D conv, I use the design and implementation from [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch), just minor modification is done to fit the network into this person reID system.

I found that learning rate has a significant impact on the final performance. Here are the learning rates I used (may not be the best): 0.0003 for temporal pooling, 0.0003 for temporal attention, 0.0001 for RNN, 0.0001 for 3D conv.

Other detailed settings for different temporal modeling could be found in `models/ResNet.py`

### Performance of the paper

| Model            | mAP |CMC-1 | CMC-5 | CMC-10 | CMC-20 |
| :--------------- | ----------: | ----------: | ----------: | ----------: | ----------: | 
| image-based      |   74.1  | 81.3 | 92.6 | 94.8 | 96.7 |
| pooling    |   75.8  | 83.1 | 92.8 | 95.3 | 96.8   |
| attention    |  76.7 | 83.3 | 93.8 | 96.0 | 97.4 |
| rnn    |   73.9 | 81.6 | 92.8 | 94.7 | 96.3 |
| 3d conv    |  70.5 | 78.5 | 90.9 | 93.9 | 95.9 |

### Performance of my result

| Model            | mAP |CMC-1 | CMC-5 | CMC-10 | CMC-20 |
| :--------------- | ----------: | ----------: | ----------: | ----------: | ----------: | 
| pooling    |   76.3  | 82.2 | 93.5 | 95.7 | 96.7   |
| attention    |  75.7 | 82.2 | 93.3 | 95.7 | 97.1 |

### Prerequisites
* Linux kernel 4.15.0-58-generic
* gcc version 5.4.0
* Ubuntu 16.04.6 LTS
* CUDA Version 9.0.176
* Python 3.6.8
* Pytorch 1.1.0

### Python Packages Version
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
asn1crypto                0.24.0                py36_1003    conda-forge
blas                      1.0                         mkl  
bzip2                     1.0.8                h516909a_0    conda-forge
ca-certificates           2019.6.16            hecc5488_0    conda-forge
cairo                     1.14.12              h8948797_3  
certifi                   2019.6.16                py36_1    conda-forge
cffi                      1.12.3           py36h2e261b9_0  
chardet                   3.0.4                 py36_1003    conda-forge
cryptography              2.7              py36h72c5cf5_0    conda-forge
cudatoolkit               9.0                  h13b8566_0  
cycler                    0.10.0                   py36_0  
cython                    0.29.12          py36he6710b0_0  
dbus                      1.13.6               h746ee38_0  
dominate                  2.3.5                      py_0    conda-forge
expat                     2.2.6                he6710b0_0  
ffmpeg                    4.0                  hcdf2ecd_0  
fontconfig                2.13.0               h9420a91_0  
freeglut                  3.0.0             hf484d3e_1005    conda-forge
freetype                  2.9.1                h8a8886c_1  
glib                      2.56.2               hd408876_0  
graphite2                 1.3.13            hf484d3e_1000    conda-forge
gst-plugins-base          1.14.0               hbbd80ab_1  
gstreamer                 1.14.0               hb453b48_1  
harfbuzz                  1.9.0             he243708_1001    conda-forge
hdf5                      1.10.2               hc401514_3    conda-forge
icu                       58.2                 h9c2bf20_1  
idna                      2.8                   py36_1000    conda-forge
intel-openmp              2019.4                      243  
jasper                    2.0.14               h07fcdf6_1  
joblib                    0.13.2                   py36_0  
jpeg                      9b                   h024ee3a_2  
kiwisolver                1.1.0            py36he6710b0_0  
libedit                   3.1.20181209         hc058e9b_0  
libffi                    3.2.1                hd88cf55_4  
libgcc-ng                 9.1.0                hdf63c60_0  
libgfortran               3.0.0                         1    conda-forge
libgfortran-ng            7.3.0                hdf63c60_0  
libglu                    9.0.0             hf484d3e_1000    conda-forge
libopencv                 3.4.2                hb342d67_1  
libopus                   1.3                  h7b6447c_0  
libpng                    1.6.37               hbc83047_0  
libsodium                 1.0.17               h516909a_0    conda-forge
libstdcxx-ng              9.1.0                hdf63c60_0  
libtiff                   4.0.10               h2733197_2  
libuuid                   1.0.3                h1bed415_2  
libvpx                    1.7.0                h439df22_0  
libxcb                    1.13                 h1bed415_1  
libxml2                   2.9.9                hea5a465_1  
matplotlib                3.1.0            py36h5429711_0  
mkl                       2019.4                      243  
mkl-service               2.0.2            py36h7b6447c_0  
mkl_fft                   1.0.12           py36ha843d7b_0  
mkl_random                1.0.2            py36hd81dba3_0  
ncurses                   6.1                  he6710b0_1  
ninja                     1.9.0            py36hfd86e86_0  
numpy                     1.16.4           py36h7e9f1db_0  
numpy-base                1.16.4           py36hde5b4d6_0  
olefile                   0.46                     py36_0  
opencv                    3.4.2            py36h6fd60c2_1  
openssl                   1.1.1c               h516909a_0    conda-forge
pandas                    0.25.0           py36he6710b0_0  
pcre                      8.43                 he6710b0_0  
pillow                    6.1.0            py36h34e0f95_0  
pip                       19.1.1                   py36_0  
pixman                    0.38.0            h516909a_1003    conda-forge
py-opencv                 3.4.2            py36hb342d67_1  
pycocotools               2.0.0            py36h470a237_0    hcc
pycparser                 2.19                     py36_0  
pyopenssl                 19.0.0                   py36_0    conda-forge
pyparsing                 2.4.0                      py_0  
pyqt                      5.9.2            py36h05f1152_2  
pysocks                   1.7.0                    py36_0    conda-forge
python                    3.6.8                h0371630_0  
python-dateutil           2.8.0                    py36_0  
pytorch                   1.1.0           py3.6_cuda9.0.176_cudnn7.5.1_0    pytorch
pytz                      2019.1                     py_0  
pyyaml                    5.1.1            py36h7b6447c_0  
pyzmq                     18.0.2           py36h1768529_2    conda-forge
qt                        5.9.7                h5867ecd_1  
readline                  7.0                  h7b6447c_5  
requests                  2.22.0                   py36_1    conda-forge
scikit-learn              0.21.2           py36hd81dba3_0  
scipy                     1.3.0            py36h7c811a0_0  
setuptools                41.0.1                   py36_0  
sip                       4.19.8           py36hf484d3e_0  
six                       1.12.0                   py36_0  
sqlite                    3.29.0               h7b6447c_0  
tk                        8.6.8                hbc83047_0  
torchfile                 0.1.0                      py_0    conda-forge
torchvision               0.3.0           py36_cu9.0.176_1    pytorch
tornado                   6.0.3            py36h7b6447c_0  
tqdm                      4.32.1                     py_0  
urllib3                   1.25.3                   py36_0    conda-forge
visdom                    0.1.8.8                       0    conda-forge
websocket-client          0.56.0                   py36_0    conda-forge
wheel                     0.33.4                   py36_0  
xorg-fixesproto           5.0               h14c3975_1002    conda-forge
xorg-inputproto           2.3.2             h14c3975_1002    conda-forge
xorg-kbproto              1.0.7             h14c3975_1002    conda-forge
xorg-libx11               1.6.8                h516909a_0    conda-forge
xorg-libxau               1.0.9                h14c3975_0    conda-forge
xorg-libxext              1.3.4                h516909a_0    conda-forge
xorg-libxfixes            5.0.3             h516909a_1004    conda-forge
xorg-libxi                1.7.10               h516909a_0    conda-forge
xorg-xextproto            7.3.0             h14c3975_1002    conda-forge
xorg-xproto               7.0.31            h14c3975_1007    conda-forge
xz                        5.2.4                h14c3975_4  
yaml                      0.1.7                had09818_2  
zeromq                    4.3.2                he1b5a44_2    conda-forge
zlib                      1.2.11               h7b6447c_3  
zstd                      1.3.7                h0b5b093_0  