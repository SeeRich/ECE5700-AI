# Course Project - Segment Anything
* This course project will revolve around Segment Anything Model (v1/v2)
* Anonymous GitHub link: https://anonymous.4open.science/r/ECE570-FINAL-D9EA/README.md

### Selected Papers:
* [Mobile SAM Paper (arXiv 2023)](https://arxiv.org/pdf/2306.14289)
* [SAM Paper (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf)
* [Tiny ViT Paper (ECCV 2022)](https://arxiv.org/abs/2207.10666)
* Removed: [ViT Paper (ICLR 2021)](https://openreview.net/pdf?id=YicbFdNTTy)

### SAM-HQ:
* [SAM HQ Paper (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/5f828e38160f31935cfe9f67503ad17c-Paper-Conference.pdf)
* [SAM HQ Github](https://github.com/SysCV/sam-hq/tree/main)

### Useful Links:
* [Mobile SAM Github](https://github.com/ChaoningZhang/MobileSAM?tab=readme-ov-file)
* [Mobile SAM Libtorch](https://github.com/cyrillkuettel/Libtorch-MobileSAM-Example)
* [Segment Anything ONNX Exporter](https://github.com/vietanhdev/samexporter)
* [Candle SAM TinyViT](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/segment_anything/tiny_vit.rs)

#### Fast SAM
* [Fast SAM Paper (arXiv 2023)](https://arxiv.org/pdf/2306.12156)
* [Fast SAM Github](https://github.com/CASIA-IVA-Lab/FastSAM)

#### Vision Transformers
* [Original ViT Paper][https://openreview.net/forum?id=YicbFdNTTy]
* [Vision Transformer (ViT) - Github](https://github.com/google-research/vision_transformer)
* [Vision Transformer Tutorial](https://www.v7labs.com/blog/vision-transformer-guide)
* [Tiny ViT - Github](https://github.com/microsoft/Cream/tree/main/TinyViT)


## Setup:
* Install pytorch with CUDA support. See requirements.txt
* Download checkpoint [file](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
* Download SA-B1 dataset

## TODO:
* Write a script that will take a directory of images and:
    1. Come up with ViT Tiny model (i.e. try using the one from pytorch or somewhere else already)
    2. Generate embedding vector from image using existing Meta SAM model
    3. Generate embedding vector from image using ViT Tiny
    4. Measure MSE between embedding vector and feed that back to train ViT Tiny