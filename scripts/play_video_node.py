#!/usr/bin/python3

import rospy
import cv_bridge
import cv2

from sensor_msgs.msg import Image
from tools.ros_tools import pack_image


class PlayVideoNode:

    def __init__(self):
        rospy.init_node('play_video')

        self.video_path = rospy.get_param('~video_path')        
        rospy.loginfo('video_path = %s', self.video_path)

        self.image_pub = rospy.Publisher('/show_video/image', Image, queue_size=10)
        self.br = cv_bridge.CvBridge()
        
        self.capture = None
        
           
    def spin(self):    
        for i in range(10):
            self.capture = cv2.VideoCapture(self.video_path)
            if self.capture.isOpened():
                fps = self.capture.get(cv2.CAP_PROP_FPS)
                rospy.loginfo(fps)
                rate = rospy.Rate(fps)
                while not rospy.is_shutdown() and self.capture.isOpened():
                    ret, frame = self.capture.read()      
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                        image_msg = pack_image(self.br, frame, encoding='rgb8')
                        self.image_pub.publish(image_msg)
                    else:
                        break
                    rate.sleep()
            else:
                rospy.loginfo("Video not open")
                exit(1)
           

def main():
    node = PlayVideoNode()
    node.spin()


if __name__ == '__main__':
    main()
