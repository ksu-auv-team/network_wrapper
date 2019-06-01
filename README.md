# jetson_nano_inference
Jetson Nano ML install scripts, automated optimization of robotics detection models, and filter-based tracking of detections

## Motivation

Installing and setting up the new Nvidia Jetson Nano was surprisingly time consuming and unintuitive. From protobuf version conflicts, to Tensorflow versions, OpenCV recompiling with GPU, models running, models optimized, and general chaos in the ranks.

This repository is KSU-AUV's set of install tools to get the Nano up and running with a convincing and scalable demo for robot-centric uses. In particular, using detection and semantic segmentation models capable at running in real-time on a robot for $100. This gives you full control of which model to run and when. 

In the repository, you'll find a few key things:

### Install of dependencies

Getting the right versions of Tensorflow, protobufs, etc and having everyone play well on the Jetson Nano platform was a big hassle. Hopefully these will help you.

This can be accomplished via `./install.sh` run in the root of this repository, where all the models are going to be installed and linked.

### Download of pretrained models for real-time detection 

Scripts to automatically download pretrained Tensorflow inference graphs and checkpoints, then optimize with TensorRT (which was found as a critical must-have to even *run* on the Nano).

TODO: Show how to use pre-trained up here for instant gratification

## Walk-through

`jetson_live_object_detection.py` is the main live object detection program. It will take no flags and run in a debug mode with printed statements about detections found and a visualization. The visualization will include the bounding boxes around an object where the line thickness is proportional to confidence. Example use to run an ssd mobilenet v1 trt optimized model in debug mode:

```
python3 jetson_live_object_detection.py ssd_mobilenet_v1_trt_graph.pb True
```

`tf_download_and_trt_model.py` will be your pretrained model savior. You're able to download pretrained models *unoptimized* from zoo and have them placed in the `./data` directory along side the ms coco labels. After download, it will run the TensorRT optimization over them and leave you with a file named `[model]_trt_graph.pb` for use. Example use:

```
tf_download_and_trt_model.py [model]
```

Model options include:
- ssd_mobilenet_v1_coco
- ssd_mobilenet_v2_coco
- ssd_inception_v2_coco

