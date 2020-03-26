### Dependencies
* caffe (with python support)
* pytorch 0.4+ (optional if you only want to convert onnx)
* onnx  

### Convert & Run
Suppose your model is already in *.onnx format  
convert onnx2caffe
>convertCaffe.py  

run example
>pfld_caffe_inference.py
### Current support operation
* Conv
* ConvTranspose
* BatchNormalization
* MaxPool
* AveragePool
* Relu
* Sigmoid
* Dropout
* Gemm (InnerProduct only)
* Add
* Mul
* Reshape
* Upsample
* Concat
* Flatten
* Transpose/Permute
* Softmax

## PS
* You need to use [onnx-simplifier](https://github.com/daquexian/onnx-simplifier) to simplify onnx model and then run convertCaffe.py to convert it into caffe model.
* You need to install [ssd-caffe](https://github.com/weiliu89/caffe/tree/ssd) and pycaffe of ssd-caffe.
