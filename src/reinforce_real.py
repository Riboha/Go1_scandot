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
from unitree_legged_msgs.msg import MotorState, LowState

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
        model_state_dict = torch.load("/home/lair99/catkin_ws/src/dog/src/second.pt", map_location=self.device)
        
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
        
        # joint position[12:24] simulation done
        self.joint_state = [ 0.0, 0.67, -1.3, 
                                0.0, 0.67, -1.3, 
                                0.0, 0.67, -1.3, 
                                0.0, 0.67, -1.3]
        # default joint angles
        self.default_joint_state = [0.1, 0.8, -1.5,
                                    -0.1, 0.8, -1.5,
                                    0.1, 1.0, -1.5,
                                    -0.1, 1.0, -1.5]
        
        self.joint_offset = [-0.0354,0.021,0.0,
                             0.0249,0.0286,-0.018,
                            -0.0408,0.0254,0.012,
                            0.0272,0.0254,-0.009]
        
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
                self.scandot.append(-0.8577)
        # robot position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_listener = tf.TransformListener()
        self.last_robot_pose = 0
        # odometry thread
        self.odometry = rospy.Timer(rospy.Duration(nsecs=3000000), self.odometry_function)
        
        # 관절 state subscriber
        self.state_sub = rospy.Subscriber("/low_state", 
                                            LowState,
                                            self.state_sub_callback,
                                            queue_size=1)
                                        
        
        # scandot subscriber
        self.scandot_sub = rospy.Subscriber("/scandot_value", 
                                            Float32MultiArray,
                                            self.scandot_callback)
        
        self.timer = rospy.Timer(rospy.Duration(nsecs=20000000), self.control)

        self.iter = 0.0
        self.i = 0
    
    def state_sub_callback(self, msg):
        joint_state = [0.0,0.0,0.0,0.0,0.0,0.0,
                                0.0,0.0,0.0,0.0,0.0,0.0]
        joint_velocity = [0.0,0.0,0.0,0.0,0.0,0.0,
                                0.0,0.0,0.0,0.0,0.0,0.0]
        # joint states
        # for i in range(4):
        #     joint_state[i*3] = msg.motorState[i*3].q
        #     joint_velocity[i*3] = msg.motorState[i*3].dq
        #     joint_state[i*3+1] = msg.motorState[i*3+1].q
        #     joint_velocity[i*3+1] = msg.motorState[i*3+1].dq
        #     joint_state[i*3+2] = msg.motorState[i*3+2].q
        #     joint_velocity[i*3+2] = msg.motorState[i*3+2].dq
        
        for i in range(4):
            joint_state[i*3] = msg.motorState[i*3].q
            joint_velocity[i*3] = msg.motorState[i*3].dq
            joint_state[i*3+1] = msg.motorState[i*3+1].q
            joint_velocity[i*3+1] = msg.motorState[i*3+1].dq
            joint_state[i*3+2] = msg.motorState[i*3+2].q
            joint_velocity[i*3+2] = msg.motorState[i*3+2].dq
        
        self.joint_state = []
        self.joint_state = joint_state
        self.joint_velocity = []
        self.joint_velocity = joint_velocity
        
        # 좌우반전
        # self.joint_state[0:3] = joint_state[3:6]
        # self.joint_state[3:6] = joint_state[0:3]
        # self.joint_state[6:9] = joint_state[9:12]
        # self.joint_state[9:12] = joint_state[6:9]
        
        # self.joint_velocity[0:3] = joint_velocity[3:6]
        # self.joint_velocity[3:6] = joint_velocity[0:3]
        # self.joint_velocity[6:9] = joint_velocity[9:12]
        # self.joint_velocity[9:12] = joint_velocity[6:9]
        
        print("joints")
        # print(self.joint_state[0:3])
        # print(self.joint_state[3:6])
        # print(self.joint_state[6:9])
        # print(self.joint_state[9:12])
        
        # print(self.joint_velocity[0:3])
        # print(self.joint_velocity[3:6])
        # print(self.joint_velocity[6:9])
        # print(self.joint_velocity[9:12])
        
        
        # projected gravity
        self.IMU_arr['imu']['quaternion'] = [msg.imu.quaternion[0], msg.imu.quaternion[1], msg.imu.quaternion[2], msg.imu.quaternion[3]]
        imu_orientation = self.IMU_arr['imu']['quaternion']
        q = torch.tensor([imu_orientation])
        self.projected_gravity_vector = self.quat_rotate_inverse(q, self.gravity_vector)
        
    def quat_rotate_inverse(self, q, v):
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
    
    def scandot_callback(self, msg):
        self.scandot = msg.data
        
    def odometry_function(self, timer):
        if self.tf_listener.canTransform(target_frame="world", source_frame="camera", time=rospy.Time(0)):
            tf_msg = self.tf_buffer.lookup_transform(
            target_frame="world", source_frame="camera", time=rospy.Time(0))
            
            self.robot_x = tf_msg.transform.translation.x
            self.robot_y = tf_msg.transform.translation.y
            self.robot_z = tf_msg.transform.translation.z
            
            odom_transform = tf.transformations.concatenate_matrices(
            tf.transformations.translation_matrix([ self.robot_x,
                                                    self.robot_y,
                                                    self.robot_z]),
            tf.transformations.quaternion_matrix([  0.0,
                                                    0.0,
                                                    tf_msg.transform.rotation.z,
                                                    tf_msg.transform.rotation.w])
            )
            tf_matrix = np.array(odom_transform)
              
    def control(self, timer):
        # 이번 action
        action = []
        joint_state = []
        joint_velocity = []
        observations = []
        scandot = []
        scandot = self.scandot
        observations = torch.zeros([235]).to(device="cuda:0")
        # joint
        joint_state = self.joint_state
        joint_velocity = self.joint_velocity
        joint_for_infer = []
        for i, joint in enumerate(joint_state):
            joint_for_infer.append(joint - self.default_joint_state[i])

        # robot linear velocity
        observations[:3] = torch.tensor([0.0 ,0.0,0.0]) * 2.0
        # robot angular velocity
        observations[3:6] = torch.tensor([0.0,0.0,0.0]) * 0.25
        # projected gravity
        # observations[6:9] = torch.tensor(self.projected_gravity_vector)
        observations[6:9] = torch.tensor([-0.0,-0.0,-0.98])
        # command
        observations[9:12] = torch.tensor([0.0,0.0,0.0]).to(device="cuda:0") * self.commands_scale
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
        # print("action")
        # action = action.clip(min=-2.0, max=2.0)
        self.prev_action = []
        self.prev_action = torch.tensor(action)
        msg = Control_12()
        send_action = action.cpu().tolist()
        msg.coltrol_12 = send_action
        # hip reduction
        # for i in range(4):
        #     msg.coltrol_12[i*3] = send_action[i*3] * 0.5
        
        # msg.coltrol_12[0:3] = send_action[3:6]
        # msg.coltrol_12[3:6] = send_action[0:3]
        # msg.coltrol_12[6:9] = send_action[9:12]
        # msg.coltrol_12[9:12] = send_action[6:9]
        
        self.pub.publish(msg)
        # print("action")
        # print(msg.coltrol_12[0:3])
        # print(msg.coltrol_12[3:6])
        # print(msg.coltrol_12[6:9])
        # print(msg.coltrol_12[9:12])
        
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