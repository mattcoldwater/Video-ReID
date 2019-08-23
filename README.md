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

### Python Packages Version (conda 4.7.10)
* cudatoolkit               9.0                  h13b8566_0  
* matplotlib                3.1.0            py36h5429711_0  
* numpy                     1.16.4           py36h7e9f1db_0  
* numpy-base                1.16.4           py36hde5b4d6_0  
* opencv                    3.4.2            py36h6fd60c2_1  
* openssl                   1.1.1c               h516909a_0    conda-forge
* pandas                    0.25.0           py36he6710b0_0  
* pillow                    6.1.0            py36h34e0f95_0  
* pip                       19.1.1                   py36_0  
* python                    3.6.8                h0371630_0  
* python-dateutil           2.8.0                    py36_0  
* pytorch                   1.1.0           py3.6_cuda9.0.176_cudnn7.5.1_0    pytorch