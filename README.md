# text-detection-ctpn

Text detection mainly based on ctpn (connectionist text proposal network).
The original paper can be found [here](https://arxiv.org/abs/1609.03605). 
The original repo in caffe can be found in [here](https://github.com/tianzhi0549/CTPN). 
For more detail about the paper and code, see this [blog](http://slade-ruan.me/2017/10/22/text-detection-ctpn/).
The original repo in tensorflow can be found in [here](https://github.com/eragonruan/text-detection-ctpn).

This repo contains modifications necessary to run text-detection-ctpn on Windows.
My god, show some love for windows users.
We are a minority deprived of any attention or care...
***
# setup
- requirements: tensorflow, cython, opencv-python, easydict
- build the library by
```shell
cd lib/utils
make.bat
```
***
# demo
- download the checkpoints from release, unzip it in checkpoints/
- put your images in data/demo, the results will be saved in data/results, and run demo in the root 
```shell
python ./ctpn/demo.py
```
# some results
<img src="/data/results/KakaoTalk_20180621_222012218.jpg" width=640 height=480 />
<img src="/data/results/KakaoTalk_20180621_222013226.jpg" width=640 height=480 />
<img src="/data/results/KakaoTalk_20180621_222014292.jpg" width=640 height=480 />
<img src="/data/results/KakaoTalk_20180621_222015318.jpg" width=640 height=480 />
***
