Notes:
- all images to jpg
- dataset folder: images, labels
- read image: PIL is faster than cv2
- resize images. All images to 1 shape, so they can fit into 1 tensor of tensors
- collate_fn puts images as a tensor of tensors (batch of images), but classes and boxes are lists of tensors
- model is taken from damo-yolo
- Adam doesn't work well with my task
- conf thresho for nms in model while training is 0.001 for correct training and mAP. confthresh from configs is used in evaluation function. During inference also can build model with conf thresh from configs
- ideally you should not have if statements in forward pass if you want to use onnx?
- onnx + cuda + dynamic input size = issues. Warmap for convs for every new input size. Can configure it though
- openvino automatically uses bf16, so it's optimized by default, no need for half precision export or inference. With that said, can be unstable on gpu. with DAMO yolo it is unstable. Works on cpu just fine. dynamic shape makes sence again only woth cpu inference, doesn't work correctly with gpu infer. So with Openvino use fixxed size, full precision, GPU infer. Alternative - dynamic input and CPU.


values.yaml
makefile memrylimit
