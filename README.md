# Stealth X-Adv:  Stealth Physical Adversarial Object Attacks against X-ray Prohibited Item Detection (On Progress)

## Introduction
Stealth X-ADV is a Adversarial Object for attack X-ray Object Detector [Faster-RCNN, SSD].  
The existing adversarial attack technique for X-ray object detectors, known as X-ADV (https://github.com/DIG-Beihang/X-adv), has demonstrated effective performance in X-ray environments. However, the generated objects are highly conspicuous, making them easily noticeable in real-world settings, which increases the risk of being detected by security personnel in advance. Therefore, this study aims to enhance the stealthiness of the original X-ADV method.  
To achieve this, we introduce two approaches: **(1) Key-ADV** and **(2) Few-Pixel Attack in X-ray**.  


<div align="center">
  <img src="https://github.com/user-attachments/assets/209f459b-1208-4c74-a096-b590a74088e2" width="400" />
</div>
<div align="center">
  (As can be seen, the original X-ADV generates objects with suspicious shapes.)
</div>  

## Methodology  
### 1) Key-ADV  

<div align="center">
  
| Faste-RCNN (OPIXray)| X-ADV | X-ADV (Changing Depth) | Stealth X-ADV (ours) | 
|----------|----------|----------|----------|
|  Missing Object | 430     | 245     | 299     |
| TP   | 1144     | 1349     | 1173     |
| FP    | 621     | 412     | 843     |
| mAP    | 0.5344     | 0.6667     | 0.5352     | 

</div>  

We conducted an attack by subtly adjusting the perceptual loss function to embed adversarial patterns into the grooves of a "key"—a common and inconspicuous item in luggage.  
As you can see 'Key-ADV' shows a comparable mAP reduction to the original X-ADV, but offers the additional advantage of being able to deceive not only object detectors but also the human eye.  

<div align="center">
  <img src="https://github.com/user-attachments/assets/24f229a8-7f86-40e7-9377-d1e5dec598d5" width="500" />
  <img src="https://github.com/user-attachments/assets/7ef2a67e-14cd-4508-87ca-34c9be814676" width="500" />
</div>

### 2) Few-Pixel Attack (On progress) 
Although Key-ADV demonstrated strong attack performance, it is limited in terms of applicable object types and poses challenges for 3D printing. Therefore, we aim to apply the existing Few-Pixel Attack as an alternative approach.
<div align="center">
  <img src="https://github.com/user-attachments/assets/25b20944-0d4e-4e5a-bbd9-a203e0bbc370" width="600" />
</div>
In this study, we apply the Few-Pixel Attack to disrupt the detection of illegal luggage items, ensuring that the altered pixels are grouped together in a compact region. These pixels are then arranged into a specific pattern, allowing the attack to be embedded into ordinary luggage items for practical deployment.

## Install

### Requirements

* Python >= 3.6

* PyTorch >= 1.8

```shell
pip install -r requirements.txt
```

### Data Preparation

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


