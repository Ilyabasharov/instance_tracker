import numpy as np
import pycocotools.mask as mask_utils

from camera_objects_msgs.msg import (
    ObjectArray,
    RLE,
)

def rle_msg_to_mask(
    rle_msg: RLE,
) -> np.array:

    mask = mask_utils.decode(
        {
            'counts': rle_msg.data,
            'size': [rle_msg.height, rle_msg.width],
        }
    ).astype(np.uint8)

    return mask

def object_array_msg_to_masks(
    object_array_msg: ObjectArray,
) -> np.array:

    masks = np.array(
        [
            rle_msg_to_mask(obj.rle)
            for obj in object_array_msg.objects
        ]
    )

    return masks

def object_array_msg_to_scores(
    object_array_msg: ObjectArray,
) -> np.array:

    scores = np.array(
        [
            obj.score
            for obj in object_array_msg.objects
        ]
    )

    return scores

def object_array_msg_to_labels(
    object_array_msg: ObjectArray,
) -> np.array:

    labels = np.array(
        [
            obj.label
            for obj in object_array_msg.objects
        ]
    )

    return labels

def decode_object_array_msg(
    object_array_msg: ObjectArray,
) -> tuple:

    masks = object_array_msg_to_masks(object_array_msg)
    scores = object_array_msg_to_scores(object_array_msg)
    labels = object_array_msg_to_labels(object_array_msg)

    return masks, scores, labels

def prepare_msg(
    object_array_msg: ObjectArray,
    prediction: list,
) -> None:

    for i, track_id in prediction:
        object_array_msg.objects[i].track_id = track_id
