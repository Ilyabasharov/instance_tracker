#!/usr/bin/python3

import os
import time
import rospy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from camera_objects_msgs.msg import ObjectArray

from scripts.utils import (
    Stats,
)
from scripts.ros.ros_msgs import (
    decode_object_array_msg,
    prepare_msg,
)

from scripts.main import Tracker


class TrackNode:

    def __init__(
        self,
    ) -> None:

        rospy.init_node(
            name='track_node',
            anonymous=False,
        )

        config_path = str(
            rospy.get_param(
                param_name='~config_path', 
                default='/home/user/catkin_ws/src/pointtrack/config.yaml',
            )
        )

        self.stats_rate = int(
            rospy.get_param(
                param_name='~stats_rate',
                default='20',
            )
        )

        queue_size = int(
            rospy.get_param(
                param_name='~queue_size',
                default='15',
            )
        )

        if not os.path.exists(config_path):
            rospy.signal_shutdown(
                reason=f'Config file {config_path} was not found.',
            )

        self.tracker = Tracker(config_path)

        self.sub = rospy.Subscriber(
            name='image',
            data_class=Image,
            callback=self.process,
            queue_size=queue_size,
        )

        self.pub = rospy.Publisher(
            name='track_ids',
            data_class=ObjectArray,
            queue_size=queue_size,
        )

        self.bridge = CvBridge()
        self.stats = Stats(
            obj_names=['inference time', 'input objects', ],
            obj_types=['sec', 'count', ],
            agg_funcs=[
                lambda x, y: x - y,
                lambda x, y: x,
            ],
        )

        self.frame_count = 0

    def process(
        self,
        image_msg: Image,
    ) -> None:

        self.frame_count += 1

        image = self.bridge.imgmsg_to_cv2(
            img_msg=image_msg,
            desired_encoding='rgb8',
        )

        self.stats.start(
            data={
                'inference time': time.perf_counter(),
                'input objects': 0, #doesnt matter because of agg func
            }
        )

        labels, masks, track_ids = self.tracker.track(
            image=image,
        )

        self.stats.stop(
            data={
                'inference time': time.perf_counter(),
                'input objects': len(track_ids),
            }
        )

        if self.stats_rate > 0 and self.frame_count % self.stats_rate == 0:
            rospy.loginfo(
                self.stats.to_str(),
            )

    @staticmethod
    def run(
    ) -> None:

        rospy.spin()

def main(
) -> None:
    
    torch.backends.cudnn.fastest = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    node = TrackNode()
    node.run()
    
if __name__ == '__main__':
    main()
