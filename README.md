# Auto Cherry-Picker: Learning from High-quality Generative Data Driven by Language
Yicheng Chen, Xiangtai Li, Yining Li, Yanhong Zeng, Jianzong Wu, Xiangyu Zhao, Kai Chen

## Updates
* **[2024/06]** Our paper [Auto Cherry-Picker: Learning from High-quality Generative Data Driven by Language](https://arxiv.org/pdf/2406.20085) is released.

## Introduction
Auto Cherry Picker is a innovative framework designed to synthesize training samples for both perception and multi-modal reasoning tasks from a simple object list in natural language. It employs a nowly designed metric, CLIS, to ensure the quality of the synthetic data.

## Main Results

### Long-tailed Instance Segmentation Benchmark
|  Method   | Backbone  | $AP_r^{mask}$ | $AP^{mask}$ | 
|  ----  | ----  | ----  | ----  |
| Mask R-CNN  | ResNet-50 | 9.3 | 21.7 | 
| Mask R-CNN w. ACP | ResNet-50 | 14.5(+5.2) | 22.8(+1.1)|
| CenterNet2 w. Copy-Paste  | Swin-B | 29.3 | 39.3 | 
| CenterNet2 w. ACP | Swin-B | 30.7(+1.4) | 39.6(+0.3)|

### Open-vocabulary Object Detection Benchmark
|  Dataset   | Method | Backbone | $AP_{novel}^{box}$ | $AP^{box}$ | 
|  ----  | ----  | ----  | ----  | ----  |
| LVIS  | Grounding-DINO | Swin-T | 31.7 | 48.7 | 
| LVIS | Grounding-DINO w. ACP | Swin-T | 33.0(+1.3) | 49.2 |
| COCO  | Grounding-DINO | Swin-T | 60.4 | 57.1 | 
| COCO | Grounding-DINO w. ACP | Swin-T | 60.8(+0.4) | 56.9 |

### Multi-modal Image-based Benchmarks
| Method | LLM Backbone | MME | GQA | 
| ----  | ----  | ----  | ----  |
| LLaVA-1.5 | Vicuna-7B | 1434.4 | 58.9 | 
| LLaVA-1.5 | Vicuna-13B | 1438.3 | 60.7 | 
| LLaVA-1.5 | LLama-3-8B | 1445.3 | 60.1 | 
| LLaVA-1.5 w. ACP | Vicuna-7B | 1514.5(+80.1) | 59.3(+0.4) | 

## Installation

### Requirements
Python 3.10

Pytorch 2.3.0

### Conda Environment Setup
```
pip install -r requirements.txt
```

### Prepare Scene Graph Generator
Download Qwen1.5-14B-Chat
```
git clone https://huggingface.co/Qwen/Qwen1.5-14B-Chat
```
You can try other LLMs as Scene Graph Generator, and add it in the `config/model_config.json`.



### Prepare Image Generator
* Step 1: Download InstanceDiffusion

```
git clone https://github.com/frank-xwang/InstanceDiffusion.git
```
* Step 2: Download model weights

Please download the pretrained InstanceDiffusion from [Hugging Face](https://huggingface.co/xudongw/InstanceDiffusion/tree/main) or [Google Drive](https://drive.google.com/drive/folders/1Jm3bsBmq5sHBnaN5DemRUqNR0d4cVzqG?usp=sharing) and [SD1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt), place them under `InstanceDiffusion/pretrained` folder.

Then, create a soft link under `ACP` folder.
```
ln -s InstanceDiffusion/pretrained ./pretrained
```
* Step 3: Download CLIP
```
git clone https://huggingface.co/openai/clip-vit-large-patch14
```

* Step 4: Download SDXL Refiner (Optional)
```
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
```
To disable SDXL, you can set `args.cascade_strength` at `infer_image.py` to `0`.

### Prepare Image Filter
Please download Qwen-VL-Chat
```
git clone https://huggingface.co/Qwen/Qwen-VL-Chat
```

### Prepare Layout Filter
Please construct example pool for CLIS-L.

Download [sim_map.json](https://drive.google.com/uc?export=download&id=1vccyYDSUhoOM17k4W1vJL8v64R7IWh9m) and [relations_one_to_one.json](https://drive.google.com/uc?export=download&id=1AhXIJNxBEwO9a6MpLTkKz8dgkItGdNSe) under `config/eval/`

### Prepare Segmentor

Download SAM model weights at [Github](https://github.com/facebookresearch/segment-anything#model-checkpoints)

```
mkdir sam
cd sam
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Quick Start

```
python inference_single_data.py
```
You can custom object list at `inputs/demo.json`. The generated images are under `images/` and the synthesis training sample is under `syn_data/`.



