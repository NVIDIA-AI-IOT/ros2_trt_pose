# ---------------------------------------------------------------------------------------
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ---------------------------------------------------------------------------------------

import torch
import torchvision.transforms as transforms
import PIL.Image
import cv2
import os
import json
import trt_pose.coco
import trt_pose.models
from trt_pose.parse_objects import ParseObjects
import torch2trt
from torch2trt import TRTModule

# Pre-process image message received from cam2image
def preprocess(image, width, height):
    global device
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    device = torch.device('cuda')
    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def load_params(base_dir, human_pose_json, model_name):
    hp_json_file = os.path.join(base_dir,human_pose_json)
    if model_name == 'resnet18':
        MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
    if model_name == 'densenet121':
        MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'

    model_weights = os.path.join(base_dir, MODEL_WEIGHTS)

    with open(hp_json_file,'r') as f:
        human_pose = json.load(f)

    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    num_parts = len(human_pose['keypoints']) # Name of the body part
    num_links = len(human_pose['skeleton']) # Need to know
    parse_objects = ParseObjects(topology)
    return num_parts, num_links, model_weights, parse_objects, topology

def load_model(base_dir, model_name, num_parts, num_links, model_weights):
    #self.get_logger().info("Model Weights are loading \n")
    if model_name == 'resnet18':
        model = trt_pose.models.resnet18_baseline_att(num_parts, 2*num_links).cuda().eval()
        model.load_state_dict(torch.load(model_weights))
        MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
        OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
        height, width = 224,224
    if model_name == 'densenet121':
        model = trt_pose.models.densenet121_baseline_att(num_parts, 2*num_links).cuda().eval()
        model.load_state_dict(torch.load(model_weights))
        MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'
        OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
        height, width = 256,256

    model_file_path = os.path.join(base_dir, OPTIMIZED_MODEL)
    if not os.path.isfile(model_file_path):
        data = torch.zeros((1,3, height, width)).cuda()
        model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
        torch.save(model_trt.state_dict(), model_file_path)
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(model_file_path))

    return model_trt, height, width

def draw_objects(image, object_counts, objects, normalized_peaks, topology):
    topology = topology
    height = image.shape[0]
    width = image.shape[1]
    count = int(object_counts[0])
    K = topology.shape[0]
    for i in range(count):
        color = (0, 255, 0)
        obj = objects[0][i]
        C = obj.shape[0]
        for j in range(C):
            k = int(obj[j])
            if k >= 0:
                peak = normalized_peaks[0][j][k]
                x = round(float(peak[1]) * width)
                y = round(float(peak[0]) * height)
                cv2.circle(image, (x, y), 3, color, 2)

        for k in range(K):
            c_a = topology[k][2]
            c_b = topology[k][3]
            if obj[c_a] >= 0 and obj[c_b] >= 0:
                peak0 = normalized_peaks[0][c_a][obj[c_a]]
                peak1 = normalized_peaks[0][c_b][obj[c_b]]
                x0 = round(float(peak0[1]) * width)
                y0 = round(float(peak0[0]) * height)
                x1 = round(float(peak1[1]) * width)
                y1 = round(float(peak1[0]) * height)
                cv2.line(image, (x0, y0), (x1, y1), color, 2)

    return image
