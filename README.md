# TDMP

This is code for our papaer "Test-time Domain-agnostic Meta-prompt Learning for Multi-source Few-shot Domain Adaptation"

## Install

* Build conda enviroment

  ```
  #create a conda environment
  conda create -y -n TDMP python=3.9

  # Activate the environment
  conda activate TDMP

  #Install torch and torchvision (refer to https://pytorch.org)
  pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 torchvision==0.16.2+cu118
  ```
* Install `dassl` enviroment. Follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install it.
* Clone this project and install requirements

  ```
  pip install -r requirements.txt
  ```

## Download Dataset

Please download the [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html) and [DomainNet](https://ai.bu.edu/M3SDA/) datasets and put them into the `data/` folder.

Please refer to the corresponding splits of [OfficeHome](https://github.com/zhengzangw/PCS-FUDA/tree/master/data/splits/office_home) and [DomainNet](https://github.com/zhengzangw/PCS-FUDA/tree/master/data/splits/domainnet) from PCS-FUDA. Put them into the `data/OfficeHome/splits/` or `data/DomainNet/splits/` folders.

## Acknowledgements

Our code is based on [CoOp](https://github.com/KaiyangZhou/CoOp), [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning) and [TPT ](https://github.com/azshue/TPT)repositories. We thank all the authors for releasing their codes and datasets.
