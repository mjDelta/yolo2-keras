# yolo2-keras

## How to use this repo.
* Train on your RBC datasets.
* Load my model to detect Red Blood Cell.
* Train step by step.


## Train on your RBC datasets
First, please download the weights I have trained here(https://drive.google.com/open?id=1-M84cxZIyVeBUfxat0MF7Wn7hmLHUmsk) and put both in a directory named "models/". Its full path should be "models/tiny_yolo_backend.h5" and "models/tiny_yolo.h5".</br>
Second, please run ```train.ipynb```

## Load my model to detect Red Blood Cell
Please run ```load_model.ipynb```</br>


## Train step by step.
Step 1：Download the tiny_yolo_backend.py here(https://drive.google.com/open?id=1-M84cxZIyVeBUfxat0MF7Wn7hmLHUmsk) and put it in a directory named "models/". Its full path should be "models/tiny_yolo_backend.h5".</br>
Step 2: In shell, run ```python train.py -c config.json```.</br>
Step 3: In shell, run ```python test.py -c config.json -w models/tiny_yolo.h5```.</br>

Here's the detected sample.</br>
![Image text](https://github.com/mjDelta/yolo2-keras/blob/master/output/00009.jpg)</br>

Please enjoy it! :)
