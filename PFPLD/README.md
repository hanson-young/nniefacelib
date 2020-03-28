# Improved-PFLD  
PFPLD : A Practical Facial Pose and Landmark Detector.
![](./results/pose_landms.png)
- 用PRNet标注人脸图像的角度数据，这比原始通过solvePNP得到效果好很多,增强了模型对pose的支持
- pfld loss收敛速度比较慢，因此对多种loss进行了对比，使用wing loss的效果更加明显。
- 改进了pfld网络结构，让关键点和姿态角度都能回归的比较好，将关键点分之任务合并到pose分之中，
这样做的好处是让其相互影响，两个任务毕竟相关性较强。  

![](./results/net.png)
#### install requirements

~~~shell
- cuda
- pytorch==0.4.1+
- onnx
- onnx-simplifier
- onnxruntime
- torchvision
- opencv-python
~~~

#### Datasets
WFLW-POSE([BaiduNetdist](https://pan.baidu.com/s/1DYxfhDtWCe1aBixUzVUyEQ) 
code：mcvt)  

#### training & testing

training :

~~~shell
$ sh train_wing.sh
~~~

testing:

~~~shell
$ python test.py
~~~

#### Convert
1. Generate onnx file
```Shell
python convert_to_onnx.py

python3 -m onnxsim ./models/onnx/checkpoint_epoch_final.onnx ./models/onnx/pfpld.onnx

```

#### Example
![](./results/1.png)  

![](./results/2.png)

![](./results/3.png)

![](./results/4.png)
#### Reference: 

paper: [A Practical Facial Landmark Detector](https://arxiv.org/pdf/1902.10859.pdf)   
code: [Tensorflow Implementation](https://github.com/guoqiangqi/PFLD)  
code: [Pytorch Implementation](https://github.com/polarisZhao/PFLD-pytorch)   
#### TODO:
- [x] training code
- [x] caffe inference
- [x] ncnn inference
- [ ] nnie inference
