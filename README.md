# Auto Cherry-Picker: Learning from High-quality Generative Data Driven by Language
Yicheng Chen, Xiangtai Li, Yining Li, Yanhong Zeng, Jianzong Wu, Xiangyu Zhao, Kai Chen

## News
* **[2024/06]** Our paper Auto Cherry-Picker: Learning from High-quality Generative Data Driven by Language is released.

## Introduction
Auto Cherry Picker is a innovative framework designed to synthesize training samples for both perception and multi-modal reasoning tasks from a simple object list in natural language. It employs a nowly designed metric, CLIS, to ensure the quality of the synthetic data.

## Main Results

### Long-tailed Instance Segmentation Benchmark
|  Method   | Backbone  | AP$_r^{mask}$ | AP$^{mask}$ | 
|  ----  | ----  | ----  | ----  |
| Mask R-CNN  | ResNet-50 | 9.3 | 21.7 | 
| Mask R-CNN w. ACP | ResNet-50 | 14.5(+5.2) | 22.8(+1.1)|
| CenterNet2 w. Copy-Paste  | Swin-B | 29.3 | 39.3 | 
| CenterNet2 w. ACP | Swin-B | 30.7(+1.4) | 39.6(+0.3)|

### Open-vocabulary Object Detection Benchmark
|  Dataset   | Method | Backbone | AP$_{novel}^{box}$ | AP$^{box}$ | 
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