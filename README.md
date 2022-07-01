# Towards Robust Video Object Segmentation with Adaptive Object Calibration (ACM Multimedia 2022)


Preview version paper of this work is available at [Arxiv]() [Please stay tuned.]

Qualitative results and comparisons with previous SOTAs are available at [YouTube](https://www.youtube.com/watch?v=3F6n7tcwWkA).

**This repo is a preview version. More details will be added later. Welcome to starts ⭐ & comments 💹 & collaboration 😀 !!**

```diff
- 2022.7.1: Repo init. Please stay tuned~
```
---



## Abstract
In the booming video era, video segmentation attracts increasing research attention in the multimedia community.

Semi-supervised video object segmentation (VOS) aims at segmenting objects in all target frames of a video, given annotated object masks of reference frames. **Most existing methods build pixel-wise reference-target correlations and then perform pixel-wise tracking to obtain target masks. Due to neglecting object-level cues, pixel-level approaches make the tracking vulnerable to perturbations, and even indiscriminate among similar objects.**

Towards **robust VOS**, the key insight is to calibrate the representation and mask of each specific object to be expressive and discriminative. Accordingly, we propose a new deep network, which can adaptively construct object representations and calibrate object masks to achieve stronger robustness.

First, we construct the object representations by applying an **adaptive object proxy (AOP) aggregation** method, where the proxies represent arbitrary-shaped segments at multi-levels for reference. 

Then, prototype masks are initially generated from the reference-target correlations based on AOP.
Afterwards, such proto-masks are further calibrated through network modulation, conditioning on the object proxy representations.
We consolidate this **conditional mask calibration** process in a progressive manner, where the object representations and proto-masks evolve to be discriminative iteratively.

Extensive experiments are conducted on the standard VOS benchmarks, YouTube-VOS-18/19 and DAVIS-17. 
Our model achieves the state-of-the-art performance among existing published works, and also exhibits significantly superior robustness against perturbations.

## Requirements
* Python3
* pytorch >= 1.4.0 
* torchvision
* opencv-python
* Pillow

You can also use the docker image below to set up your env directly. However, this docker image may contain some redundent packages.

```latex
docker image: xxiaoh/vos:10.1-cudnn7-torch1.4_v3
```

A more light-weight version can be created by modified the [Dockerfile](https://github.com/JerryX1110/RPCMVOS/blob/main/Dockerfile) provided.

## Preparation
* Datasets

    * **YouTube-VOS**

        A commonly-used large-scale VOS dataset.

        [datasets/YTB/2019](datasets/YTB/2019): version 2019, download [link](https://drive.google.com/drive/folders/1BWzrCWyPEmBEKm0lOHe5KLuBuQxUSwqz?usp=sharing). `train` is required for training. `valid` (6fps) and `valid_all_frames` (30fps, optional) are used for evaluation.

        [datasets/YTB/2018](datasets/YTB/2018): version 2018, download [link](https://drive.google.com/drive/folders/1bI5J1H3mxsIGo7Kp-pPZU8i6rnykOw7f?usp=sharing). Only `valid` (6fps) and `valid_all_frames` (30fps, optional) are required for this project and used for evaluation.

    * **DAVIS**

        A commonly-used small-scale VOS dataset.

        [datasets/DAVIS](datasets/DAVIS): [TrainVal](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) (480p) contains both the training and validation split. [Test-Dev](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip) (480p) contains the Test-dev split. The [full-resolution version](https://davischallenge.org/davis2017/code.html) is also supported for training and evaluation but not required.

* pretrained weights for the backbone

  [resnet101-deeplabv3p](https://drive.google.com/file/d/1H3yUShfPqzxSt-nHJP-zYSbbu2o_RQQu/view)


## Training

* More details will be added soon.

## Inference

* More details will be added soon.

* For evaluation, please use official YouTube-VOS servers ([2018 server](https://competitions.codalab.org/competitions/19544) and [2019 server](https://competitions.codalab.org/competitions/20127)), official [DAVIS toolkit](https://github.com/davisvideochallenge/davis-2017) (for Val), and official [DAVIS server](https://competitions.codalab.org/competitions/20516#learn_the_details) (for Test-dev).



## Limitation & Directions for further exploration towards Robust VOS!

* More diverse perturbation/corruption types
* Adversial attack and defence for VOS models
* VOS model robustness verification and theoretical analysis
* Model enhancement from the perspective of data management

(to be continued...)

## Citation
If you find this work is useful for your research, please consider citing:

 ```latex
   @article{PLACEHOLDER,
     title={Towards Robust Video Object Segmentation with Adaptive Object Calibration},
     author={Xu, Xiaohao and Wang, Jinglu and Ming, Xiang and Lu, Yan},
     journal={ACM Multimedia},
     year={2022}
   }
```  

    
## Credit

**CFBI**: <https://github.com/z-x-yang/CFBI>

**Deeplab**: <https://github.com/VainF/DeepLabV3Plus-Pytorch>

**GCT**: <https://github.com/z-x-yang/GCT>

## Related Works in VOS
**Semisupervised video object segmentation repo/paper link:**

**SWEM [CVPR 2022]**:<https://tianyu-yang.com/resources/swem.pdf>

**RDE [CVPR 2022]**:<https://arxiv.org/pdf/2205.03761.pdf>

**COVOS [CVPR 2022]** :<https://github.com/kai422/CoVOS>

**RPCM [AAAI 2022 Oral]** :<https://github.com/JerryX1110/RPCMVOS>

**AOT [NeurIPS 2021]**: <https://github.com/z-x-yang/AOT>

**STCN [NeurIPS 2021]**: <https://github.com/hkchengrex/STCN>

**JOINT [ICCV 2021]**: <https://github.com/maoyunyao/JOINT>

**HMMN [ICCV 2021]**: <https://github.com/Hongje/HMMN>

**DMN-AOA [ICCV 2021]**: <https://github.com/liang4sx/DMN-AOA>

**MiVOS [CVPR 2021]**: <https://github.com/hkchengrex/MiVOS>

**SSTVOS [CVPR 2021 Oral]**: <https://github.com/dukebw/SSTVOS>

**GraphMemVOS [ECCV 2020]**: <https://github.com/carrierlxk/GraphMemVOS>

**AFB-URR [NeurIPS 2020]**: <https://github.com/xmlyqing00/AFB-URR>

**CFBI [ECCV 2020]**: <https://github.com/z-x-yang/CFBI>

**FRTM-VOS [CVPR 2020]**: <https://github.com/andr345/frtm-vos>

**STM [ICCV 2019]**: <https://github.com/seoungwugoh/STM>

**FEELVOS [CVPR 2019]**: <https://github.com/kim-younghan/FEELVOS>

(The list may be incomplete, feel free to contact me by pulling a issue and I'll add them on!)

## Useful websites for VOS
**The 1st Large-scale Video Object Segmentation Challenge**: <https://competitions.codalab.org/competitions/19544#learn_the_details>

**The 2nd Large-scale Video Object Segmentation Challenge - Track 1: Video Object Segmentation**: <https://competitions.codalab.org/competitions/20127#learn_the_details>

**The Semi-Supervised DAVIS Challenge on Video Object Segmentation @ CVPR 2020**: <https://competitions.codalab.org/competitions/20516#participate-submit_results>

**DAVIS**: <https://davischallenge.org/>

**YouTube-VOS**: <https://youtube-vos.org/>

**Papers with code for Semi-VOS**: <https://paperswithcode.com/task/semi-supervised-video-object-segmentation>

## Q&A

## Acknowledgement ❤️
Firstly, the author would like to thank Rex for his insightful viewpoints about VOS during e-mail discussion!
Also, this work is built upon CFBI. Thanks to the author of CFBI to release such a wonderful code repo for further work to build upon!

## Welcome to comments and discussions!!
Xiaohao Xu: <xxh11102019@outlook.com>

## License
This project is released under the Mit license. See [LICENSE](LICENSE) for additional details.
