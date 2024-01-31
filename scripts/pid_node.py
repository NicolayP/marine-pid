#!/usr/bin/env python
import rospy
import numpy as np
from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Empty

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Twist

from tf.transformations import quaternion_multiply, quaternion_inverse, quaternion_from_euler

import tf

from pid_class import PIDClass


rospy.init_node("PID_DP_CONTROLLER")


def q_to_matrix(q):
    """Convert quaternion into orthogonal rotation matrix.
    
    > *Input arguments*
    
    * `q` (*type:* `numpy.array`): Quaternion vector as 
    `(qx, qy, qz, qw)`
    
    > *Returns*
    
    `numpy.array`: Rotation matrix
    """
    e1 = q[0]
    e2 = q[1]
    e3 = q[2]
    eta = q[3]
    R = np.array([[1 - 2 * (e2**2 + e3**2),
                    2 * (e1 * e2 - e3 * eta),
                    2 * (e1 * e3 + e2 * eta)],
                    [2 * (e1 * e2 + e3 * eta),
                    1 - 2 * (e1**2 + e3**2),
                    2 * (e2 * e3 - e1 * eta)],
                    [2 * (e1 * e3 - e2 * eta),
                    2 * (e2 * e3 + e1 * eta),
                    1 - 2 * (e1**2 + e2**2)]])
    return R    

def q_to_euler(q):
    """`numpy.array`: Orientation error in Euler angles."""
    e1 = q[0]
    e2 = q[1]
    e3 = q[2]
    eta = q[3]
    rot = np.array([[1 - 2 * (e2**2 + e3**2),
                        2 * (e1 * e2 - e3 * eta),
                        2 * (e1 * e3 + e2 * eta)],
                    [2 * (e1 * e2 + e3 * eta),
                        1 - 2 * (e1**2 + e3**2),
                        2 * (e2 * e3 - e1 * eta)],
                    [2 * (e1 * e3 - e2 * eta),
                        2 * (e2 * e3 + e1 * eta),
                        1 - 2 * (e1**2 + e2**2)]])
    # Roll
    roll = np.arctan2(rot[2, 1], rot[2, 2])
    # Pitch, treating singularity cases
    den = np.sqrt(1 - rot[2, 1]**2)
    pitch = - np.arctan(rot[2, 1] / max(0.001, den))
    # Yaw
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    return np.array([roll, pitch, yaw])

class PIDNode(object):
    def __init__(self):
        self._Kp = np.zeros(shape=(6, 6))
        self._Ki = np.zeros(shape=(6, 6))
        self._Kd = np.zeros(shape=(6, 6))
        self._int = np.zeros(6)
        self._error_pose = np.zeros(6)

        self._current_goal_idx = 0
        self._goal_list = []

        self._namespace = rospy.get_namespace()

        if rospy.has_param('~Kp'):
            Kp_diag = rospy.get_param('~Kp')
            if len(Kp_diag) == 6:
                self._Kp = np.diag(Kp_diag)
            else:
                raise rospy.ROSException('Kp matrix error: 6 coefficients '
                                         'needed')

        rospy.loginfo('Kp=' + str([self._Kp[i, i] for i in range(6)]))

        if rospy.has_param('~Kd'):
            Kd_diag = rospy.get_param('~Kd')
            if len(Kd_diag) == 6:
                self._Kd = np.diag(Kd_diag)
            else:
                raise rospy.ROSException('Kd matrix error: 6 coefficients '
                                         'needed')

        rospy.loginfo('Kd=' + str([self._Kd[i, i] for i in range(6)]))

        if rospy.has_param('~Ki'):
            Ki_diag = rospy.get_param('~Ki')
            if len(Ki_diag) == 6:
                self._Ki = np.diag(Ki_diag)
            else:
                raise rospy.ROSException('Ki matrix error: 6 coefficients '
                                         'needed')

        rospy.loginfo('Ki=' + str([self._Ki[i, i] for i in range(6)]))

        if rospy.has_param('~waypoints'):
            self._goal_list = np.array(rospy.get_param("~waypoints")["goals"])
            rospy.loginfo('goal list=\n' + str(self._goal_list))

        if rospy.has_param('~inertial_frame_id'):
            self._inertial_frame_id = rospy.get_param("~inertial_frame_id")

        self._odom_is_init = False
        
        self._pose = {}
        self._errors = {}
        self._error_pose_prev = np.zeros(6)

        # Time step
        self._dt = 0
        self._period = 0.1
        self._prev_time = rospy.get_time()

        self._pid = PIDClass(self._Kp, self._Ki, self._Kd)
        # Publish cost representation in rviz.
        self._vis = rospy.Publisher(
            "pid/waypoints/",
            Marker,
            queue_size=10
        )

        # 6 publishers for P errors
        # 6 publishers for I errors
        # 6 publishers for D errors
        names = ["x", "y", "z", "roll", "pitch", "yaw",
                 "I_x", "I_y", "I_z", "I_roll", "I_pitch", "I_yaw",
                 "D_x", "D_y", "D_z", "D_roll", "D_pitch", "D_yaw",]

        self._error_pid1 = [rospy.Publisher(
            "pid1/errors/%s" % n,
            Float32,
            queue_size=100
        ) for n in names]

        # 6 publishers for P errors
        # 6 publishers for I errors
        # 6 publishers for D errors

        axis = ["x", "y", "z", "roll", "pitch", "yaw"]

        self._pid1_output = [rospy.Publisher(
            "pid1/output/%s" % n,
            Float32,
            queue_size=10
        ) for n in axis]

        # self._thrust_pub = rospy.Publisher(
        #     'thruster_input', WrenchStamped, queue_size=1)

        self._twist_pub = rospy.Publisher(
            '/bluerov2/cmd_vel', Twist, queue_size=1)

        # Subscribe to odometry topic
        self._odom_topic_sub = rospy.Subscriber(
            'odom', numpy_msg(Odometry), self._odometry_callback)
        
        start_service = rospy.Subscriber('/log_data/start_log', Empty, self.start_logging)
        self.started = False

    def start_logging(self, msg):
        rospy.loginfo("Start PID")
        self.started = True

    def _odometry_callback(self, msg):
        if not self.started:
            return
        """Odometry topic subscriber callback function."""
        # The frames of reference delivered by the odometry seems to be as
        # follows
        # position -> world frame
        # orientation -> world frame
        # linear velocity -> world frame
        # angular velocity -> world frame

        if self._inertial_frame_id != msg.header.frame_id:
            raise rospy.ROSException('The inertial frame ID used by the '
                                     'vehicle model does not match the '
                                     'odometry frame ID, vehicle=%s, odom=%s' %
                                     (self._inertial_frame_id,
                                      msg.header.frame_id))

        # Update the velocity vector
        # Update the pose in the inertial frame
        self._pose['pos'] = np.array([msg.pose.pose.position.x,
                                      msg.pose.pose.position.y,
                                      msg.pose.pose.position.z])

        # Using the (x, y, z, w) format for quaternions
        self._pose['rot'] = np.array([msg.pose.pose.orientation.x,
                                      msg.pose.pose.orientation.y,
                                      msg.pose.pose.orientation.z,
                                      msg.pose.pose.orientation.w])
        self._rotBtoI = q_to_matrix(self._pose['rot'])
        self._rotItoB = self._rotBtoI.T
        # Linear velocity on the INERTIAL frame
        lin_vel = np.array([msg.twist.twist.linear.x,
                            msg.twist.twist.linear.y,
                            msg.twist.twist.linear.z])
        # Transform linear velocity to the BODY frame
        lin_vel = np.dot(self._rotItoB, lin_vel)
        # Angular velocity in the INERTIAL frame
        ang_vel = np.array([msg.twist.twist.angular.x,
                            msg.twist.twist.angular.y,
                            msg.twist.twist.angular.z])
        # Transform angular velocity to BODY frame
        ang_vel = np.dot(self._rotItoB, ang_vel)
        # Store velocity vector
        self._vel = np.hstack((lin_vel, ang_vel))

        if not self._odom_is_init:
            self._goal_world_list= self._compute_world_goal(self._goal_list, self._pose)
            print(self._goal_world_list)
            self._odom_is_init = True


        if not self._update_time_step():
            return
        self._update_error()
        self._update_goal()
        self._update_vis()
        self._update_controller()
        self.publish_errors()

    def _compute_world_goal(self, goal_list, pose):
        rot = q_to_matrix(pose['rot'])
        quat = pose['rot']
        trans = pose['pos']
        T = np.zeros(shape=(4, 4))
        T[0:3, 0:3] = rot
        T[0:3, 3] = trans
        T[3, 3] = 1.
        goals = []
        for goal in goal_list:
            goal_pos_homo = np.array([goal[0], goal[1], goal[2], 1.])
            goal_quat = goal[3:]

            world_pos = (T @ goal_pos_homo)[:3]
            world_ori = quaternion_multiply(pose["rot"], goal_quat)
            world_goal = np.concatenate([world_pos, world_ori], axis=-1)
            goals.append(world_goal)
        return goals

    def _update_time_step(self):
        """Update time step."""
        t = rospy.get_time()
        self._dt = t - self._prev_time
        if self._dt < self._period:
            return False
        self._prev_time = t
        return True

    def _update_goal(self):
        pos_dist = np.linalg.norm(self._errors['pos'])
        ang_dist = np.linalg.norm(self._errors["rpy"])
        if pos_dist < 0.1 and ang_dist < 0.2:
            rospy.loginfo("Goal Reached")
            if len(self._goal_world_list) > self._current_goal_idx + 1:        
                self._current_goal_idx += 1

    def _update_error(self):
        if not self._odom_is_init:
            return
        goal = self._goal_world_list[self._current_goal_idx]

        goal_pos = goal[:3]
        pos = self._pose['pos']

        goal_quat = goal[3:7]
        quat = self._pose['rot']

        vel = self._vel
        # Express the error in body frame
        self._errors['pos'] = np.dot(self._rotItoB, goal_pos-pos)
        

        self._errors['rpy'] = q_to_euler(quaternion_multiply(quaternion_inverse(quat), goal_quat))

        # print("pos:       ", pos)
        # print("goal_pos:  ", goal_pos)
        # print("pos_error: ", self._errors['pos'])
        # print("rot:       ", q_to_euler(quat))
        # print("goal rot:  ", q_to_euler(goal_quat))
        # print("rot_error: ", self._errors['rpy'])
        
        self._errors['deriv'] = -vel
        self._errors['prop'] = np.hstack((self._errors['pos'], self._errors['rpy']))

        self._error_vel = self._errors['deriv']

    def _update_vis(self):
        if not self._odom_is_init:
            return
        m = Marker()
        m.id = 0
        m.header.frame_id = self._inertial_frame_id
        m.type = m.LINE_STRIP
        m.action = m.ADD
        m.scale.x = 0.05
        m.scale.y = 0.05
        m.color.a = 1.
        m.color.r = 1.
        m.color.g = 0.5
        m.color.b = 0.0

        m.pose.orientation.w = 1.0
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.0

        p = Point()
        p.x = self._pose['pos'][0]
        p.y = self._pose['pos'][1]
        p.z = self._pose['pos'][2]
        m.points.append(p)
        for t in self._goal_world_list[self._current_goal_idx:]:
            p = Point()
            p.x = t[0]
            p.y = t[1]
            p.z = t[2]
            m.points.append(p)
        self._vis.publish(m)

    def _update_controller(self):
        forces, self._int = self._pid.update_pid(self._errors, self._dt)
        self.publish_twist(forces)
        #self.publish_contorl_wrench(forces)

    def publish_contorl_wrench(self, force):
        """Publish the thruster manager control set-point.
        
        > *Input arguments*
        
        * `force` (*type:* `numpy.array`): 6 DoF control 
        set-point wrench vector
        """
        if not self._odom_is_init:
            return

        force_msg = WrenchStamped()
        force_msg.header.stamp = rospy.Time.now()
        force_msg.header.frame_id = '%s/%s' % (self._namespace, "base_link")
        force_msg.wrench.force.x = force[0]
        force_msg.wrench.force.y = force[1]
        force_msg.wrench.force.z = force[2]

        force_msg.wrench.torque.x = force[3]
        force_msg.wrench.torque.y = force[4]
        force_msg.wrench.torque.z = force[5]

        self._thrust_pub.publish(force_msg)

    def publish_twist(self, force):
        """Publish the thruster manager control set-point.
        
        > *Input arguments*
        
        * `force` (*type:* `numpy.array`): 6 DoF control 
        set-point wrench vector
        """
        if not self._odom_is_init:
            return

        twist = Twist()
        twist.linear.x = force[0]
        twist.linear.y = force[1]
        twist.linear.z = force[2]

        twist.angular.x = force[3]
        twist.angular.y = force[4]
        twist.angular.z = force[5]
        #print(twist)
        self._twist_pub.publish(twist)

    def publish_errors(self):
        if not self._odom_is_init:
            return
        # PID 1 (external)
        for p, e in zip(self._error_pid1, np.concatenate([np.dot(self._Kp, self._errors['prop']),
                                                          np.dot(self._Ki, self._int),
                                                          np.dot(self._Kd, self._errors['deriv'])])):
            p.publish(e)
        
        for p, o in zip(self._pid1_output, np.dot(self._Kp, self._errors['prop']) + \
                                           np.dot(self._Ki, self._int) + \
                                           np.dot(self._Kd, self._errors['deriv'])):
            p.publish(o)

if __name__ == "__main__":
    try:
        node = PIDNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Caught exception")
    print("Exiting")