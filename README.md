# 2025-DL-HW1
Course homework. Professor: C C,Hsu

* Dataset
  * Dataset URL: https://cchsu.info/files/images.zip
  * The mini-ImageNet dataset is a smaller version of the ImageNet dataset, created to reduce the computational complexity while retaining the essential properties of the larger dataset.
  * Download the dataset and then put the folder "images" at the root of the repository.

* Pre-trained ckpt
   * All pretrained checkpoints of Q1 and Q2 are under "output" folder.

* Quick Start
```
git clone https://github.com/chingheng0808/2025-DL-HW1.git
cd 2025-DL-HW1
## For sure you have create a environment installed [Pytorch](https://pytorch.org/get-started/previous-versions/).
## Q1
# train baseline (ResNet34)
python train1.py --q1 baseline --model_name org_resnet34
# train preprocessing ver. 
python train1.py --q1 anyInput --model_name anyInput

## Q2
# python train2.py --model_name small_net
```
