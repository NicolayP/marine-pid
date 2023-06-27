#!/usr/bin/env python

import rospy
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float32

rospy.init_node("FIX_RVIZ")

def forward_error_dx(msg):
    pub_d_x.publish(msg)

def forward_error_dy(msg):
    pub_d_y.publish(msg)

def forward_error_dz(msg):
    pub_d_z.publish(msg)

def forward_error_dpitch(msg):
    pub_d_roll.publish(msg)

def forward_error_droll(msg):
    pub_d_pitch.publish(msg)

def forward_error_dyaw(msg):
    pub_d_yaw.publish(msg)



sub_d_x = rospy.Subscriber("pid1/errors/D_x", numpy_msg(Float32), forward_error_dx)
sub_d_y = rospy.Subscriber("pid1/errors/D_y", numpy_msg(Float32), forward_error_dy)
sub_d_z = rospy.Subscriber("pid1/errors/D_z", numpy_msg(Float32), forward_error_dz)
sub_d_roll = rospy.Subscriber("pid1/errors/D_roll", numpy_msg(Float32), forward_error_droll)
sub_d_pitch = rospy.Subscriber("pid1/errors/D_pitch", numpy_msg(Float32), forward_error_dpitch)
sub_d_yaw = rospy.Subscriber("pid1/errors/D_yaw", numpy_msg(Float32), forward_error_dyaw)


pub_d_x = rospy.Publisher("pid1_fix/errors/D_x", Float32, queue_size=10)
pub_d_y = rospy.Publisher("pid1_fix/errors/D_y", Float32, queue_size=10)
pub_d_z = rospy.Publisher("pid1_fix/errors/D_z", Float32, queue_size=10)
pub_d_roll = rospy.Publisher("pid1_fix/errors/D_roll", Float32, queue_size=10)
pub_d_pitch = rospy.Publisher("pid1_fix/errors/D_pitch", Float32, queue_size=10)
pub_d_yaw = rospy.Publisher("pid1_fix/errors/D_yaw", Float32, queue_size=10)


if __name__ == "__main__":
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Caught exception")
    print("Exiting")