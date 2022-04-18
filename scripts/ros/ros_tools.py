import numpy as np
import torch
from typing import List
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from ros_yolact_edge_segmentation.msg import InstancesArray, Instance, Mask


def pack_image(cv_bridge: CvBridge, image: np.ndarray, encoding: str='rgb8') -> Image:
    image_msg = cv_bridge.cv2_to_imgmsg(image, encoding=encoding)
    return image_msg


def unpack_image_msg(cv_bridge: CvBridge, image_msg: Image, encoding: str='rgb8') -> np.ndarray:
    image = cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding=encoding)
    return image


def pack_masks(masks: torch.Tensor, classes: torch.Tensor, image_msg: Image) -> Mask:
    MAX_DETS = 1000
    
    out = Mask()
    out.height = image_msg.height
    out.width = image_msg.width

    if torch.numel(masks) != 0:        
        with_zeros = torch.cat((torch.zeros(1, masks.shape[1], masks.shape[2]), masks), dim=0)
        result_mask = torch.argmax(with_zeros, dim=0)
        for i in range(1, len(classes) + 1):
            result_mask[result_mask==i] = result_mask[result_mask==i] + MAX_DETS * (classes[i - 1] + 1)
        result = result_mask.cpu().detach().numpy().astype(np.uint32)
        out.data = result.flatten()
    else:
        out.data = np.zeros(out.height * out.width, np.uint32)
    
    return out


def pack_boxes_for_transfer(model_output: List, image_msg_header: Header) -> InstancesArray:
    classes, scores, boxes, masks = model_output
    out = InstancesArray()
    out.header = image_msg_header
     
    for box, score, class_id, mask in zip(boxes, scores, classes, masks):
        detection = Instance()

        detection.score = score
        detection.class_id = class_id

        detection.box_y = int(box[1])
        detection.box_x = int(box[0])
        detection.box_height = int(box[3] - box[1])
        detection.box_width = int(box[2] - box[0])
        
        out.objects.append(detection)
    return out


