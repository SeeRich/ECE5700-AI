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

# ECE570-FINAL

## Instructions and notable files:
* Project Overview Python Notebook: **final_project.ipynb**
    * This is the main code to run and showcase the trained model performance. However, keep in mind, if running on CPU backend, you will need at least 16 GB of RAM to run the original SAM model. Also keep in mind, the 2.6 GB pretrained weights may take a while to download for the original SAM model.
* Short overview video: **ProjectOverview.mp4**
* Model Training Code: **src/train.py**
    * Note: the training code was run on a powerful Nvidia CUDA GPU for over 12 hours. This wasn\'t tested with any other Pytorch backends. An example command is provided at the top of the file for training on 220000 source images.
* Python requirements: **frozen_requirements.txt**

### Usage in Google Colab (preferred method)
1. Download two files: **archive.zip** and **final_project.ipynb**
2. Open the python notebook in Colab like usual (prefer GPU enabled runtime)
3. Upload the archive.zip using the file upload button on the left hand side near the refresh directory button.
4. Run the first cell which will detect the Colab environment and unzip the source archive.
5. Continue running the cells as normal.

### Usage on local machine (Linux/Mac tested):
* It is advised to create a python virtual environment to avoid package version incompatibilities. This can be done with (Linux):
```shell
python -m venv ./venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -r frozen_requirements.txt
```

### Generate archive.zip:
```shell
zip archive -@ < archive.txt
# To test
mkdir archive_unzipped
unzip archive.zip -d archive_unzipped
```