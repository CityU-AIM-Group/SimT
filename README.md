# SimT: Handling Open-set Noise for Domain Adaptive Semantic Segmentation

by [Xiaoqing Guo](https://guo-xiaoqing.github.io/).

## Summary:

### Intoduction:
This repository is for our CVPR 2022 paper ["SimT: Handling Open-set Noise for Domain Adaptive Semantic Segmentation"]()

### Framework:
![](https://github.com/CityU-AIM-Group/SimT/blob/main/network.png)

## Usage:
### Requirement:
Pytorch 1.3 & Pytorch 1.7 are ok
Python 3.6

### Preprocessing:
Clone the repository:
```
git clone https://github.com/Guo-Xiaoqing/SimT.git
cd SimT 
bash sh_warmup.sh ## Stage of warmup
bash sh_simt.sh ## Stage of training with SimT
```

### Data preparation:
The pseudo labels generated from the UDA black box of BAPA-Net [1] can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1lehNWiG6YPkJd5EvXl8-qgoT5iemJ-Y9?usp=sharing)

The pseudo labels generated from the SFDA black box of SFDASeg [2] can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1oi98NhGngXCoCQPhJ9IpX_GvRY1XgA2R?usp=sharing)

[1] Yahao Liu, Jinhong Deng, Xinchen Gao, Wen Li, and Lixin Duan. Bapa-net: Boundary adaptation and prototype align- ment for cross-domain semantic segmentation. In ICCV, pages 8801–8811, 2021.
[2] Jogendra Nath Kundu, Akshay Kulkarni, Amit Singh,Varun Jampani, and R Venkatesh Babu. Generalize then adapt: Source-free domain adaptive semantic segmentation. In ICCV, pages 7046–7056, 2021.

### Pretrained model:
You should download the pretrained model, warmup UDA model, and warmup SFDA model from [Google Drive](https://drive.google.com/drive/folders/1A6207JIEHx0tfaRNeJpEnYvfILX9HkpA?usp=sharing), and then put them in the './snapshots' folder for initialization. 

### Well trained model:
You could download the well trained UDA and SFDA models from [Google Drive](https://drive.google.com/drive/folders/1A6207JIEHx0tfaRNeJpEnYvfILX9HkpA?usp=sharing).

## Log file
Log file can be found [here](https://github.com/CityU-AIM-Group/SimT/blob/main/logs/)

## Citation:
```
@inproceedings{guo2022simt,
  title={SimT: Handling Open-set Noise for Domain Adaptive Semantic Segmentation},
  author={Guo, Xiaoqing and Liu, Jie and Liu, Tongliang and Yuan, Yixuan},
  booktitle= {CVPR},
  year={2022}
}
```

## Questions:
Please contact "xiaoqingguo1128@gmail.com" 
