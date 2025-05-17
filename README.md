# Stealth X-Adv:  Stealth Physical Adversarial Object Attacks against X-ray Prohibited Item Detection

<div align="center">
  <img src="https://github.com/user-attachments/assets/24f229a8-7f86-40e7-9377-d1e5dec598d5" width="800" />
  <img src="https://github.com/user-attachments/assets/7ef2a67e-14cd-4508-87ca-34c9be814676" width="800" />
</div>


## Introduction
Stealth X-ADV is a Adversarial Object for attack X-ray Object Detector.  
The existing adversarial attack technique for X-ray object detectors, known as X-ADV (https://github.com/DIG-Beihang/X-adv), has demonstrated effective performance in X-ray environments. However, the generated objects are highly conspicuous, making them easily noticeable in real-world settings, which increases the risk of being detected by security personnel in advance. Therefore, this study aims to enhance the stealthiness of the original X-ADV method. To achieve this, we introduce two approaches: (1) Key-ADV and (2) Few-Pixel Attack in X-ray.  


<div align="center">
  <img src="https://github.com/user-attachments/assets/209f459b-1208-4c74-a096-b590a74088e2" width="400" />
</div>
(As can be seen, the original X-ADV generates objects with suspicious shapes.)

## Methodology  
1) Key-ADV
| Faste-RCNN (OPIXray)| X-ADV (Original) | X-ADV (Changing Depth) | Stealth X-ADV | 
|----------|----------|----------|----------|
|  Missing  | 430     | 245     | 299     |
| TP   | 1144     | 1349     | 1173     |
| FP    | 621     | 412     | 843     |
| mAP    | 0.5344     | 0.6667     | 0.5352     |  

Stealth X-ADV shows a comparable mAP reduction to the original X-ADV, but offers the additional advantage of being able to deceive not only object detectors but also the human eye.  
2)Few-Pixel Attack

## Install

### Requirements

* Python >= 3.6

* PyTorch >= 1.8

```shell
pip install -r requirements.txt
```

### Data Preparation

#### XAD

The XAD dataset will be released after accepted.

#### OPIXray & HiXray

Please refer to [**this website**](https://github.com/DIG-Beihang/XrayDetection) to acquire download links.

#### Data Structure

The downloaded data should look like this:

```
dataset_root
|-- train
|      |-- train_annotation
|      |-- train_image
|      |-- train_knife.txt
|-- test
       |-- test_annotation
       |-- test_image
       |-- test_knife.txt
```

After acquiring the datasets, you should modify `data/config.py` to set the dataset directory.

### VOC pretrained weights

For SSD detection models, the pre-trained weight on VOC0712 can be found at [here](https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth).

For Faster R-CNN models, we apply the pre-trained weight from [this issue](https://github.com/chenyuntc/simple-faster-rcnn-pytorch/issues/63), which does not need to be converted from caffe.

## Usage

### Training

Training for SSD models (original, DOAM, LIM):

```shell
python train_ssd.py --dataset OPIXray/HiXray/XAD \
    --model_arch original/DOAM/LIM \
    --transfer ./weights/ssd300_mAP_77.43_v2.pth \
    --save_folder ./save
```

Training for Faster R-CNN:

```shell
python train_frcnn.py --dataset OPIXray/HiXray/XAD \
    --transfer ./weights/vgg16-397923af.pth \
    --save_folder ./save
```

### Attack

Attack SSD models (original, DOAM, LIM) with X-Adv:

```shell
python attack_ssd.py --dataset OPIXray/HiXray/XAD \
    --model_arch original/DOAM/LIM \
    --ckpt_path ../weights/model.pth \
    --patch_place reinforce \
    --patch_material iron \
    --save_path ./results
```

Attack Faster R-CNN with X-Adv:

```shell
python attack_frcnn.py --dataset OPIXray/HiXray/XAD \
    --patch_place reinforce \
    --ckpt_path ../weights/model.pth \
    --patch_material iron \
    --save_path ./results
```

Below are some combinations of `patch_place` and `patch_material`:

| Method            | `patch_place` | `patch_material` |
| ----------------- | ------------- | ---------------- |
| meshAdv           | fix           | iron_fix         |
| AdvPatch          | fix_patch     | iron             |
| X-Adv | reinforce     | iron             |

### Evaluation

Evaluate SSD models (original, DOAM, LIM):

```shell
python test_ssd.py --dataset OPIXray/HiXray/XAD \
    --model_arch original/DOAM/LIM \
    --ckpt_path ../weights/model.pth \
    --phase path/to/your/adver_image
```

Evaluate Faster R-CNN:

```shell
python test_frcnn.py --dataset OPIXray/HiXray/XAD \
    --ckpt_path ../weights/model.pth \
    --phase path/to/your/adver_image
```

## Citation

If this work helps your research, please cite the following paper.

```
@inproceedings{liu2023xadv,
  title={X-Adv: Physical Adversarial Object Attacks against X-ray Prohibited Item Detection},
  author={Liu, Aishan and Guo, Jun and Wang, Jiakai and Liang, Siyuan and Tao, Renshuai and Zhou, Wenbo and Liu, Cong and Liu, Xianglong and Tao, Dacheng},
  booktitle={32st USENIX Security Symposium (USENIX Security 23)},
  year={2022}
}
```

## Reference

[Original implementation and pre-trained weight of SSD](https://github.com/amdegroot/ssd.pytorch)

[Implementation and pre-trained weight of Faster R-CNN](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)

[Official repository of DOAM and OPIXray](https://github.com/DIG-Beihang/OPIXray)

[Official repository of LIM and HiXray](https://github.com/HiXray-author/HiXray)
