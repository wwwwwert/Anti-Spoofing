# AS project 

## Project description

In this project RawNet2 ([paper](https://arxiv.org/pdf/2011.01108.pdf)) model was implemented for Anti Spoofing task.

## Project structure
- **/hw_as** - project scripts
- _install_dependencies.sh_ - script for dependencies installation
- _requirements.txt_ - Python requirements list
- _train.py_ - script to run train
- _test.py_ - script to run test

## Installation guide

It is strongly recommended to use new virtual environment for this project.

To install all required dependencies and final model run:
```shell
./install_dependencies.sh
```

This script installs Python requirements and downloads necessary folders:
- **/best_model** - best model state dict and config
- **/custom_test_data** - custom test dataset
- **/test_results** - tests results _json_ files

_Logical Access_ dataset is automatically downloaded when the associated dataset is used.

## Reproduce results
To run train with _Logical Access train_ dataset:
```shell
python -m train -c hw_as/configs/train_config.json
```

To run test inference with _Logical Access eval_ dataset:
```shell
python test.py \
   -c hw_as/configs/test_la_eval.json \
   -r best_model/best_model.pth \
   -o la_eval_result.json \
   -b 50 \
   --calculate_scores
```

To run test inference with custom dataset you will need to create a directory with the following structure:
- _custom_dataset/_
   - _spoofed/_ - put spoofed recordings here
   - _original/_ - put legit recordings here

Run:
```shell
python test.py \
   -c hw_as/configs/test_config.json \
   -r best_model/best_model.pth \
   -o custom_data_result.json \
   -b 1 \
   -t custom_dataset/ \
   --calculate_scores
```

To run inference without labeling create the following directory:
- _custom_dataset/spoofed_ - put all recordings here

Run test without _calculate_scores_ argument:
```shell
python test.py \
   -c hw_as/configs/test_config.json \
   -r best_model/best_model.pth \
   -o testing_result.json \
   -b 1 \
   -t custom_dataset/ 
```


## Author
Dmitrii Uspenskii HSE AMI 4th year.
