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
import time

class scandot_manager:

    def __init__(self):
        rospy.Subscriber('lio_sam/mapping/map_local', PointCloud2, self.cloud_callback)
        rospy.Subscriber('lio_sam/mapping/odometry', Odometry, self.pose_callback)
        
        self.pub = rospy.Publisher("/scandot", PointCloud2, queue_size=1)
        self.pub_test = rospy.Publisher("/new_points", PointCloud2, queue_size=1)
        
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
        self.transformed_points = None
        
        # scandot parameters
        self.grid_scale = 0.1
        self.rostime = None
        
        self.timer = rospy.Timer(rospy.Duration(nsecs=20000), self.result_cloud)
        
        
    def result_cloud(self, timer):
        # print("start")
        # scandot 배열 (x,y)
        # (0,0) ~ (17,10)
        scandots = np.full((17, 11), -5.0)
        if self.transformed_points != None:
            for x, y, z in self.transformed_points:
                # 천장 필터링
                if z > 1.5:
                    continue
                # grid 매기기
                grid_x = int(((self.grid_scale*8.5 + x) / self.grid_scale))
                grid_y = int(((self.grid_scale*5.5 + y) / self.grid_scale))
                if (grid_x > 16 or grid_x < 0 or
                    grid_y > 10 or grid_y < 0):
                    continue
                
                if scandots[grid_x][grid_y] < z:
                    scandots[grid_x][grid_y] = z
        
            # points로 바꾸기
            points_ = []
            for i in range(17):
                for j in range(11):
                    if scandots[i][j] != -5.0:
                        points_.append([(i-8.5)*self.grid_scale,
                                    (j-5.5)*self.grid_scale,
                                    scandots[i][j]])

            a = PointCloud2()
            a.header.frame_id = 'base_link'
            scandot = pc2.create_cloud_xyz32(a.header, points_)
            self.pub.publish(scandot)
            
            b = PointCloud2()
            b.header.frame_id = 'base_link'
            filtered_pc_msg = pc2.create_cloud_xyz32(b.header, self.transformed_points)
            self.pub_test.publish(filtered_pc_msg)
            self.transformed_points = None
            
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

    def cut_cloud_z(self, cloud, xmin, xmax, ymin, ymax, zmin, zmax):
        filter_axis = 'x'
        axis_min = xmin
        axis_max = xmax
        cloud = self.do_passthrough(cloud, filter_axis, axis_min, axis_max)

        filter_axis = 'y'
        axis_min = ymin
        axis_max = ymax
        cloud = self.do_passthrough(cloud, filter_axis, axis_min, axis_max)
        
        filter_axis = 'z'
        axis_min = zmin
        axis_max = zmax
        cloud = self.do_passthrough(cloud, filter_axis, axis_min, axis_max)
        
        return cloud

    def cloud_callback(self, input_ros_msg):
        # print(input_ros_msg.header)
        
        cloud = self.ros_to_pcl(input_ros_msg)    
        cloud = self.cut_cloud_z(cloud,   self.robot_x-2.0,  self.robot_x+2.0, 
                                        self.robot_y-2.0,  self.robot_y+2.0,
                                        self.robot_z-2.0,  self.robot_z+0.3)
        ros_msg = self.pcl_to_ros(cloud, "map")
        self.raw_map = ros_msg
        

    def pose_callback(self, pose_msg):
        # position
        self.robot_x = pose_msg.pose.pose.position.x
        self.robot_y = pose_msg.pose.pose.position.y
        self.robot_z = pose_msg.pose.pose.position.z
        
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
        prev = time.time()
        i = 0
        # 로봇 pose에 맞게 map transform
        transformed_points_ = []
        for p in pc2.read_points(self.raw_map, skip_nans=True):
            point = [p[0], p[1], p[2], 1.0]
            transformed_point = np_odom_inv.dot(point)[:3]
            transformed_points_.append(transformed_point)
            i+=1
        self.transformed_points = transformed_points_
        print("transform 소요시간 : ", time.time() - prev)
        print(i)

if __name__ == "__main__":
    rospy.init_node('scandot', anonymous=True)

    a = scandot_manager()

    rospy.spin()