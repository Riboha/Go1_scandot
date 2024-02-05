import rospy
from unitree_legged_msgs.msg import Control_12
from sensor_msgs.msg import JointState
import tf2_ros
import tf
import numpy as np
import time
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import ModelStates
from unitree_legged_msgs.msg import MotorState

import torch
import torch.nn as nn
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        activation = nn.ELU()
        self.device = "cuda:0"
        mlp_input_dim_a = 235
        actor_hidden_dims = [512, 256,128]
        num_actions = 12
        
        actor_layers = [] #행동결정
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers).to(device=self.device)
        
        # model 불러오기
        model_state_dict = torch.load("/home/lair99/catkin_ws/src/dog/src/reinforce_model.pt", map_location=self.device)
        
        # key 이름 바꾸기, 불필요한거 삭제
        for i in range(4):
            model_state_dict['model_state_dict'][str(2*i)+".weight"] = model_state_dict['model_state_dict']["actor."+str(2*i)+".weight"]
            model_state_dict['model_state_dict'][str(2*i)+".bias"] = model_state_dict['model_state_dict']["actor."+str(2*i)+".bias"]
            del model_state_dict['model_state_dict']["actor."+str(2*i)+".weight"]
            del model_state_dict['model_state_dict']["actor."+str(2*i)+".bias"]
            del model_state_dict['model_state_dict']["critic."+str(2*i)+".weight"]
            del model_state_dict['model_state_dict']["critic."+str(2*i)+".bias"]
            
        del model_state_dict['model_state_dict']["std"]

        self.actor.load_state_dict(model_state_dict['model_state_dict'])
        self.actor.eval()
        
        self.std = nn.Parameter(1.0 * torch.ones(num_actions)).to(self.device)
        self.distribution = None
        Normal.set_default_validate_args = False
        
        print(f"Actor MLP: {self.actor}")

    def act(self, observations):
        with torch.no_grad():
            return self.actor(observations)

        

class control_publisher:

    def __init__(self):

        self.pub = rospy.Publisher("/reinforce_control", Control_12, queue_size=1)
        
        self.actor = Actor()
        
        self.robot_x = 0
        self.robot_y = 0
        self.robot_z = 0
        
        self.robot_linear_velocity = [0.0, 0.0, 0.0]
        self.robot_angular_velocity = [0.0, 0.0, 0.0]
        
        # for gravity vector
        self.projected_gravity_vector = [0.0,0.0,0.0]
        self.gravity_vector = torch.tensor([[0.0,0.0,-1.0]])
        self.IMU_arr = {
            'imu': {
                'quaternion': torch.zeros(4),
                'gyroscope': torch.zeros(3),
                'accelerometer': torch.zeros(3)
            }
        }
        
        # True : 반전
        self.left_right = False
        
        # joint position[12:24] simulation done
        self.joint_state = [ 0.0, 0.67, -1.3, 
                                0.0, 0.67, -1.3, 
                                0.0, 0.67, -1.3, 
                                0.0, 0.67, -1.3]
        # default joint angles
        
        
        if self.left_right:
            self.default_joint_state = [0.1, 0.8, -1.5,
                                        -0.1, 0.8, -1.5,
                                        0.1, 1.0, -1.5,
                                        -0.1, 1.0, -1.5]
        else:
            self.default_joint_state = [-0.1, 0.8, -1.5,
                                    0.1, 0.8, -1.5,
                                    -0.1, 1.0, -1.5,
                                    0.1, 1.0, -1.5]
        
        # previous joint position
        self.prev_joint_state = [ 0.0, 0.67, -1.3, 
                                0.0, 0.67, -1.3, 
                                0.0, 0.67, -1.3, 
                                0.0, 0.67, -1.3]
        # joint velocity [24:36]
        self.joint_velocity = [0.0,0.0,0.0,0.0,0.0,0.0,
                                0.0,0.0,0.0,0.0,0.0,0.0]
        self.prev_time = 0.0

        # previous action [36:48]
        self.prev_action = torch.tensor([ 0.0, 0.67, -1.3, 
                                0.0, 0.67, -1.3, 
                                0.0, 0.67, -1.3, 
                                0.0, 0.67, -1.3]).to(device="cuda:0")
        
        self.commands_scale = torch.tensor([2.0, 2.0, 0.25], device="cuda:0")
        
        # scandot []
        self.scandot = []
        for i in range(187):
                self.scandot.append(0.0)
        # robot position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_listener = tf.TransformListener()
        self.last_robot_pose = 0
        # odometry thread
        self.odometry = rospy.Timer(rospy.Duration(nsecs=3000000), self.odometry_function)
        
        # 관절 state subscriber
        self.joint_state_sub1 = rospy.Subscriber("/go1_gazebo/FL_hip_controller/state", 
                                            MotorState,
                                            self.joint_callback1)
        self.joint_state_sub2 = rospy.Subscriber("/go1_gazebo/FL_thigh_controller/state", 
                                            MotorState,
                                            self.joint_callback2)
        self.joint_state_sub3 = rospy.Subscriber("/go1_gazebo/FL_calf_controller/state", 
                                            MotorState,
                                            self.joint_callback3)
        self.joint_state_sub4 = rospy.Subscriber("/go1_gazebo/FR_hip_controller/state", 
                                            MotorState,
                                            self.joint_callback4)
        self.joint_state_sub5 = rospy.Subscriber("/go1_gazebo/FR_thigh_controller/state", 
                                            MotorState,
                                            self.joint_callback5)
        self.joint_state_sub6 = rospy.Subscriber("/go1_gazebo/FR_calf_controller/state", 
                                            MotorState,
                                            self.joint_callback6)
        self.joint_state_sub7 = rospy.Subscriber("/go1_gazebo/RL_hip_controller/state", 
                                            MotorState,
                                            self.joint_callback7)
        self.joint_state_sub8 = rospy.Subscriber("/go1_gazebo/RL_thigh_controller/state", 
                                            MotorState,
                                            self.joint_callback8)
        self.joint_state_sub9 = rospy.Subscriber("/go1_gazebo/RL_calf_controller/state", 
                                            MotorState,
                                            self.joint_callback9)
        self.joint_state_sub10 = rospy.Subscriber("/go1_gazebo/RR_hip_controller/state", 
                                            MotorState,
                                            self.joint_callback10)
        self.joint_state_sub11 = rospy.Subscriber("/go1_gazebo/RR_thigh_controller/state", 
                                            MotorState,
                                            self.joint_callback11)
        self.joint_state_sub12 = rospy.Subscriber("/go1_gazebo/RR_calf_controller/state", 
                                            MotorState,
                                            self.joint_callback12)
                                        
        
        # scandot subscriber
        self.scandot_sub = rospy.Subscriber("/scandot_value", 
                                            Float32MultiArray,
                                            self.scandot_callback)
        # imu subscriber
        self.sub = rospy.Subscriber("/gazebo/model_states", ModelStates ,self.imuCallback)
        
        self.timer = rospy.Timer(rospy.Duration(nsecs=20000000), self.control)

        self.iter = 0.0
        self.i = 0
    
    def joint_callback1(self, msg):
        # FL hip
        self.joint_state[0] = msg.q
        self.joint_velocity[0] = msg.dq
    def joint_callback2(self, msg):
        # FL hip
        self.joint_state[1] = msg.q
        self.joint_velocity[1] = msg.dq
    def joint_callback3(self, msg):
        # FL hip
        self.joint_state[2] = msg.q
        self.joint_velocity[2] = msg.dq
    def joint_callback4(self, msg):
        # FL hip
        self.joint_state[3] = msg.q
        self.joint_velocity[3] = msg.dq
    def joint_callback5(self, msg):
        # FL hip
        self.joint_state[4] = msg.q
        self.joint_velocity[4] = msg.dq
    def joint_callback6(self, msg):
        # FL hip
        self.joint_state[5] = msg.q
        self.joint_velocity[5] = msg.dq
    def joint_callback7(self, msg):
        # FL hip
        self.joint_state[6] = msg.q
        self.joint_velocity[6] = msg.dq
    def joint_callback8(self, msg):
        # FL hip
        self.joint_state[7] = msg.q
        self.joint_velocity[7] = msg.dq
    def joint_callback9(self, msg):
        # FL hip
        self.joint_state[8] = msg.q
        self.joint_velocity[8] = msg.dq
    def joint_callback10(self, msg):
        # FL hip
        self.joint_state[9] = msg.q
        self.joint_velocity[9] = msg.dq
    def joint_callback11(self, msg):
        # FL hip
        self.joint_state[10] = msg.q
        self.joint_velocity[10] = msg.dq
    def joint_callback12(self, msg):
        # FL hip
        self.joint_state[11] = msg.q
        self.joint_velocity[11] = msg.dq
    
    def quat_rotate_inverse(q, v):
        #q = torch.tensor(q)
        shape = q.shape
        q_w = q[:, 0]
        q_vec = q[:, 1:4]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * \
            torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
                shape[0], 3, 1)).squeeze(-1) * 2.0
        return a - b + c
    
    def imuCallback(self, msg):
        
        # self.IMU_arr['imu']['quaternion'] = [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
        # self.IMU_arr['imu']['gyroscope'] = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        # self.IMU_arr['imu']['accelerometer'] = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        
        self.IMU_arr['imu']['quaternion'] = [msg.pose[2].orientation.w, msg.pose[2].orientation.x, msg.pose[2].orientation.y, msg.pose[2].orientation.z]
        
        imu_orientation = self.IMU_arr['imu']['quaternion']
        q = torch.tensor([imu_orientation])
        #print(self.IMU_arr['imu']['quaternion'])
        #print(self.IMU_arr['imu']['gyroscope'])
        #print(self.IMU_arr['imu']['accelerometer'])
        
        self.projected_gravity_vector = control_publisher.quat_rotate_inverse(q, self.gravity_vector)
        
        self.robot_linear_velocity = [0.0, 0.0, 0.0]
        self.robot_angular_velocity = [0.0, 0.0, 0.0]
        
        self.robot_linear_velocity = [msg.twist[2].linear.x, 
                                      msg.twist[2].linear.y,
                                      msg.twist[2].linear.z]
        self.robot_angular_velocity = [msg.twist[2].angular.x,
                                       msg.twist[2].angular.y,
                                       msg.twist[2].angular.z]
        # print(self.projected_gravity_vector)
        
    def scandot_callback(self, msg):
        self.scandot = msg.data
        
    def odometry_function(self, timer):
        # if self.tf_listener.canTransform(target_frame="world", source_frame="camera", time=rospy.Time(0)):
        #     tf_msg = self.tf_buffer.lookup_transform(
        #     target_frame="world", source_frame="camera", time=rospy.Time(0))
            
        #     self.robot_x = tf_msg.transform.translation.x
        #     self.robot_y = tf_msg.transform.translation.y
        #     self.robot_z = tf_msg.transform.translation.z
            
        #     odom_transform = tf.transformations.concatenate_matrices(
        #     tf.transformations.translation_matrix([ self.robot_x,
        #                                             self.robot_y,
        #                                             self.robot_z]),
        #     tf.transformations.quaternion_matrix([  0.0,
        #                                             0.0,
        #                                             tf_msg.transform.rotation.z,
        #                                             tf_msg.transform.rotation.w])
        #     )
        #     tf_matrix = np.array(odom_transform)
        pass

              
    def control(self, timer):
        # 이번 action
        action = []
        joint_state = []
        joint_velocity = []
        observations = []
        scandot = []
        observations = torch.zeros([235]).to(device="cuda:0")
        # joint
        
        scandot = self.scandot
        
        if not self.left_right:
            joint_state[0:3] = self.joint_state[3:6]
            joint_state[3:6] = self.joint_state[0:3]
            joint_state[6:9] = self.joint_state[9:12]
            joint_state[9:12] = self.joint_state[6:9]
            
            joint_velocity[0:3] = self.joint_velocity[3:6]
            joint_velocity[3:6] = self.joint_velocity[0:3]
            joint_velocity[6:9] = self.joint_velocity[9:12]
            joint_velocity[9:12] = self.joint_velocity[6:9]
        else:
            joint_state = self.joint_state
            joint_velocity = self.joint_velocity
        
        
        joint_for_infer = []
        # print("joints")
        # print(joint_state[0:3])
        # print(joint_state[3:6])
        # print(joint_state[6:9])
        # print(joint_state[9:12])
        for i, joint in enumerate(joint_state):
            joint_for_infer.append(joint - self.default_joint_state[i])

        # robot linear velocity
        observations[:3] = torch.tensor(self.robot_linear_velocity) * 2.0
        # robot angular velocity
        observations[3:6] = torch.tensor(self.robot_angular_velocity) * 0.25
        # projected gravity
        # observations[6:9] = torch.tensor(self.projected_gravity_vector)
        # observations[6:9] = torch.tensor([-0.0159,-0.1026,-0.9946])
        observations[6:9] = torch.tensor([-0.0,-0.0,-0.98])
        # command
        observations[9:12] = torch.tensor([0.1 ,0.0,0.0]).to(device="cuda:0") * self.commands_scale
        # joint position
        observations[12:24] = torch.tensor(joint_for_infer)
        # joint velocity [rad/s]
        observations[24:36] = torch.tensor(joint_velocity) * 0.05
        # previous action
        observations[36:48] = torch.tensor(self.prev_action)
        # scandots
        observations[48:235] = torch.tensor(scandot)

        # print(observations[48:235])
        ####### model input #######
        action = self.actor.act(observations)
        
        # action[0:3] = action_[3:6]
        # action[3:6] = action_[0:3]
        # action[6:9] = action_[9:12]
        # action[9:12] = action_[6:9]
        self.prev_action = []
        self.prev_action = torch.tensor(action)
        msg = Control_12()
        send_action = []
        send_action = action.cpu().tolist()
    
        if self.left_right:
            msg.coltrol_12[0:3] = send_action[3:6]
            msg.coltrol_12[3:6] = send_action[0:3]
            msg.coltrol_12[6:9] = send_action[9:12]
            msg.coltrol_12[9:12] = send_action[6:9]
        else:
            msg.coltrol_12 = send_action
        
        for i in range(4):
            msg.coltrol_12[i*3] = msg.coltrol_12[i*3] * 0.5
        
        # print("action")
        # print(msg.coltrol_12[0:3])
        # print(msg.coltrol_12[3:6])
        # print(msg.coltrol_12[6:9])
        # print(msg.coltrol_12[9:12])
        
        self.pub.publish(msg)
        
        if self.i == 0:
            self.iter += 0.01
        else:
            self.iter -= 0.01
            
        if self.iter > 0.3:
            self.i=1
        if self.iter < 0.0:
            self.i=0
        

if __name__ == "__main__":
    rospy.init_node('reinforce_model_infer', anonymous=True)

    a = control_publisher()

    rospy.spin()