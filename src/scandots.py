import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Transform, TransformStamped
import sensor_msgs.point_cloud2 as pc2
import pcl
import struct
import tf
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

class scandot_manager:

    def __init__(self):
        rospy.Subscriber('lio_sam/mapping/map_local', PointCloud2, self.cloud_callback)
        rospy.Subscriber('lio_sam/mapping/odometry', Odometry, self.pose_callback)
        
        self.pub = rospy.Publisher("/new_points", PointCloud2, queue_size=1)
        
        self.robot_x = 0
        self.robot_y = 0
        self.robot_z = 0
        self.robot_twist_x = 0
        self.robot_twist_y = 0
        self.robot_twist_z = 0
        self.robot_twist_w = 0
        self.tf_listener = tf.TransformListener()
        # 원본 map pointcloud
        self.raw_map = PointCloud2()
        self.xyz = np.array([[0,0,0]])
        self.filtered_pc = None
        self.odom_transform = None
        # 로봇 transform
        self.robot_transform = TransformStamped()
        
        self.timer = rospy.Timer(rospy.Duration(nsecs=10000), self.result_cloud)
        
    def result_cloud(self, timer):

        # input_ros_msg = do_transform_cloud(self.raw_map, self.robot_transform)
        # cloud = self.ros_to_pcl(self.raw_map)    
        # cloud = self.cut_cloud(cloud,   self.robot_x-5.0,  self.robot_x+5.0, 
        #                                 self.robot_y-5.0,  self.robot_y+5.0)
        # ros_msg = self.pcl_to_ros(cloud, "map")
        # self.pub.publish(ros_msg)
        # print("publish")
        
        ## base_link에 맞춰 변환
        ## 로봇 시점에 맞춰 pointcloud transform
        if self.odom_transform is not None:
            transformed_points = []
            for p in pc2.read_points(self.raw_map, skip_nans=True):
                point = [p[0], p[1], p[2], 1.0]
                transformed_point = self.odom_transform.dot(point)[:3]
                transformed_points.append(transformed_point)
            
            ## transform된 pointcloud 시각화
            a = PointCloud2()
            a.header.frame_id = 'base_link'
            filtered_pc_msg = pc2.create_cloud_xyz32(a.header, transformed_points)
            self.pub.publish(filtered_pc_msg)
            
            # cloud = self.ros_to_pcl(filtered_pc_msg)
            # cloud = self.cut_cloud(cloud,   -5.0,  5.0, 
            #                                 -5.0,  5.0)
            # ros_msg = self.pcl_to_ros(cloud, "base_link")
            # self.pub.publish(ros_msg)
            
    def ros_to_pcl(self, ros_cloud):
        points_list = []

        for data in pc2.read_points(ros_cloud, skip_nans=True):
            points_list.append([data[0], data[1], data[2], data[3]])
        

        pcl_data = pcl.PointCloud_PointXYZRGB()
        pcl_data.from_list(points_list)

        return pcl_data
    
    def pcl_to_ros(self, pcl_array, frameid):
    
        ros_msg = PointCloud2()

        ros_msg.header.stamp = rospy.Time.now()
        ros_msg.header.frame_id = frameid

        ros_msg.height = 1
        ros_msg.width = pcl_array.size

        ros_msg.fields.append(PointField(
                                name="x",
                                offset=0,
                                datatype=PointField.FLOAT32, count=1))
        ros_msg.fields.append(PointField(
                                name="y",
                                offset=4,
                                datatype=PointField.FLOAT32, count=1))
        ros_msg.fields.append(PointField(
                                name="z",
                                offset=8,
                                datatype=PointField.FLOAT32, count=1))
        ros_msg.fields.append(PointField(
                                name="rgb",
                                offset=16,
                                datatype=PointField.FLOAT32, count=1))

        ros_msg.is_bigendian = False
        ros_msg.point_step = 32
        ros_msg.row_step = ros_msg.point_step * ros_msg.width * ros_msg.height
        ros_msg.is_dense = False
        buffer = []

        for data in pcl_array:
            s = struct.pack('>f', data[3])
            i = struct.unpack('>l', s)[0]
            # pack = ctypes.c_uint32(i).value

            # r = (pack & 0x00FF0000) >> 16
            # g = (pack & 0x0000FF00) >> 8
            # b = (pack & 0x000000FF)

            buffer.append(struct.pack('ffffBBBBIII', data[0], data[1], data[2], 1.0, 100, 100, 100, 0, 0, 0, 0))

        ros_msg.data = b"".join(buffer)

        return ros_msg
    
    def do_passthrough(self, pcl_data,filter_axis,axis_min,axis_max):
        passthrough = pcl_data.make_passthrough_filter()
        passthrough.set_filter_field_name(filter_axis)
        passthrough.set_filter_limits(axis_min, axis_max)
        return passthrough.filter()

    def cut_cloud(self, cloud, xmin, xmax, ymin, ymax):
        filter_axis = 'x'
        axis_min = xmin
        axis_max = xmax
        cloud = self.do_passthrough(cloud, filter_axis, axis_min, axis_max)

        filter_axis = 'y'
        axis_min = ymin
        axis_max = ymax
        cloud = self.do_passthrough(cloud, filter_axis, axis_min, axis_max)
        return cloud

    def cloud_callback(self, input_ros_msg):
        print(input_ros_msg.header)
        # self.xyz = np.array([[0,0,0]])
        # # rgb = np.array([[0,0,0]])
        # #self.lock.acquire()
        # gen = pc2.read_points(input_ros_msg, skip_nans=True)
        # int_data = list(gen)

        # for x in int_data:
        #     test = x[3] 
        #     # cast float32 to int so that bitwise operations are possible
        #     s = struct.pack('>f' ,test)
        #     # i = struct.unpack('>l',s)[0]
        #     # you can get back the float value by the inverse operations
        #     # pack = ctypes.c_uint32(i).value
        #     # r = (pack & 0x00FF0000)>> 16
        #     # g = (pack & 0x0000FF00)>> 8
        #     # b = (pack & 0x000000FF)
        #     # prints r,g,b values in the 0-255 range
        #                 # x,y,z can be retrieved from the x[0],x[1],x[2]
        #     self.xyz = np.append(self.xyz,[[x[0],x[1],x[2]]], axis = 0)
        #     # rgb = np.append(rgb,[[r,g,b]], axis = 0)
           
        self.raw_map = input_ros_msg
        

    def pose_callback(self, pose_msg):
        # position
        self.robot_x = pose_msg.pose.pose.position.x
        self.robot_y = pose_msg.pose.pose.position.y
        self.robot_z = pose_msg.pose.pose.position.z
        # orientation
        self.robot_twist_x = pose_msg.pose.pose.orientation.x
        self.robot_twist_y = pose_msg.pose.pose.orientation.y
        self.robot_twist_z = pose_msg.pose.pose.orientation.z
        self.robot_twist_w = pose_msg.pose.pose.orientation.w
        
        quaternion = [pose_msg.pose.pose.orientation.x,
                      pose_msg.pose.pose.orientation.y,
                      pose_msg.pose.pose.orientation.z,
                      pose_msg.pose.pose.orientation.w]
        
        a = quaternion[0]**2 + quaternion[1]**2 + quaternion[2]**2 + quaternion[3]**2
        quaternion_inverse = [quaternion[0]/a, 
                              -quaternion[1]/a,
                              -quaternion[2]/a,
                              -quaternion[3]/a,]
        
        # robot transform(반대로)
        self.robot_transform.transform.translation.x = -pose_msg.pose.pose.position.x
        self.robot_transform.transform.translation.y = -pose_msg.pose.pose.position.y
        self.robot_transform.transform.translation.z = -pose_msg.pose.pose.position.z
        self.robot_transform.transform.rotation.x = pose_msg.pose.pose.orientation.x
        self.robot_transform.transform.rotation.y = pose_msg.pose.pose.orientation.y
        self.robot_transform.transform.rotation.z = pose_msg.pose.pose.orientation.z
        self.robot_transform.transform.rotation.w = pose_msg.pose.pose.orientation.w
        
        self.odom_transform = tf.transformations.concatenate_matrices(
            tf.transformations.translation_matrix([pose_msg.pose.pose.position.x,
                                                    pose_msg.pose.pose.position.y,
                                                    pose_msg.pose.pose.position.z]),
            tf.transformations.quaternion_matrix([pose_msg.pose.pose.orientation.x,
                                                    pose_msg.pose.pose.orientation.y,
                                                    pose_msg.pose.pose.orientation.z,
                                                    pose_msg.pose.pose.orientation.w])
        )
        np_odom = np.array(self.odom_transform)
        np_odom_inv = np.linalg.inv(np_odom)
        self.odom_transform = np_odom_inv
        

if __name__ == "__main__":
    rospy.init_node('tutorial', anonymous=True)

    a = scandot_manager()

    rospy.spin()