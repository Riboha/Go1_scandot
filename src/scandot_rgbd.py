import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
import sensor_msgs.point_cloud2 as pc2
import pcl
import struct
import tf
import tf2_ros
import time
from std_msgs.msg import Float32MultiArray

class scandot_manager:

    def __init__(self):
        rospy.Subscriber('/pose_graph/octree', PointCloud2, self.cloud_callback)
        
        # 시각화용 topic
        self.pub = rospy.Publisher("/scandot", PointCloud2, queue_size=1)
        self.pub_test = rospy.Publisher("/new_points", PointCloud2, queue_size=1)
        
        #실제 사용할 scandot value
        self.pub_scandot_value = rospy.Publisher("/scandot_value", Float32MultiArray, queue_size=1)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        initial_grid_corners_ = []
        for i in range(17): # x
            for j in range(11): # y
                initial_grid_corners_.append([-0.5 + j*0.1, -0.8 + i*0.1, 0.0, 1.0])
        self.initial_grid_corners = np.array(initial_grid_corners_)
        self.grid_corners = []
        
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
        self.cropped_cloud = 0
        self.cropped_cloud_pcl = 0
        # 로봇 transform
        self.robot_transform = TransformStamped()
        self.transformed_points = None
        # scandot parameters
        self.grid_scale = 0.1
        self.rostime = None
        self.scandot_base = []
        self.kdtree = 0
        self.init_trigger = False
        self.init_trigger2 = 0
        # odometry thread
        self.odometry = rospy.Timer(rospy.Duration(nsecs=3000000), self.odometry_function)
        # crop thread
        self.cropper = rospy.Timer(rospy.Duration(nsecs=10000000), self.crop_cloud)
        # scandot thread
        self.scandot_th = rospy.Timer(rospy.Duration(nsecs=15000000), self.scandot_cal)
        
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
            
            self.scandot_base = tf_matrix.dot(self.initial_grid_corners.T)[:3,:].T
    
    def crop_cloud(self, timer):
        # map point cloud 자르기
        cloud = self.ros_to_pcl(self.raw_map)    
        cloud = self.cut_cloud_z(cloud,   self.robot_x-1.3,  self.robot_x+1.3, 
                                        self.robot_y-1.3,  self.robot_y+1.3,
                                        self.robot_z-2.0,  self.robot_z+0.3)
        ## 시각화 목적 ,없애도됨
        self.cropped_cloud = self.pcl_to_ros(cloud, "world")
        self.pub_test.publish(self.cropped_cloud)
        
        cloud_arr = []
        ## kdtree로 저장(z축 0으로 만들고)
        for point in cloud:
            cloud_arr.append([point[0],point[1],0.0])
        cloud_np = np.array(cloud_arr, dtype="float32")
        if len(cloud_arr) > 1:
            # prev = time.time()
            z_zero = pcl.PointCloud()
            z_zero.from_array(cloud_np)   
            self.kdtree = z_zero.make_kdtree_flann()  
            self.cropped_cloud_pcl = cloud
            # print("crop hz : ", 1/(time.time() - prev))
    def scandot_cal(self, timer):
        if self.cropped_cloud_pcl != 0:
            try:
                prev = time.time()
                scandot_base = self.scandot_base
                scandots = []       # 사용할 scandot (robot frame)
                scandots_map = []   # 시각화용 scandot (map frame)
                kdtree = self.kdtree
                cropped_cloud_pcl = self.cropped_cloud_pcl
                # 값이 할당되지 않은 포인트 수
                no_point = 0
                for i, scandot_point in enumerate(scandot_base):
                    z = []
                    # scandot의 한 점
                    searchPoint = pcl.PointCloud()
                    searchPoints = np.zeros((1,3), dtype=np.float32)
                    searchPoints[0][0] = scandot_point[0]
                    searchPoints[0][1] = scandot_point[1]
                    searchPoints[0][2] = scandot_point[2]
                    searchPoint.from_array(searchPoints)
                    
                    # x,y 평면상에서 가장 가까운 점 5개 뽑기
                    [ind, sqdist] = kdtree.nearest_k_search_for_cloud(searchPoint, 5)
                    
                    if True:
                        for j in range(5):
                            z.append(cropped_cloud_pcl[ind[0][j]][2])
                        z.sort()
                        z_max = z[2]
                    else:
                        z_max = 1.0
                        no_point += 1

                    # 시작지점 고려(로봇 바로 밑 point가 없는 경우)
                    if self.init_trigger == False: 
                        scandots.append(-self.robot_z-0.45)
                        scandots_map.append([scandot_point[0],
                                            scandot_point[1], 
                                            -0.45])
                    else:   # 맵 충분히 만들어짐
                        scandots.append(-self.robot_z + z_max)
                        scandots_map.append([scandot_point[0], 
                                            scandot_point[1], 
                                            z_max])

                if self.init_trigger == False and no_point == 0:
                    self.init_trigger2 += 1
                    if self.init_trigger2 > 10:
                        self.init_trigger = True

                a = Float32MultiArray()
                a.data = scandots
                self.pub_scandot_value.publish(a)

                b = PointCloud2()
                b.header.frame_id = 'world'
                scandot_msg = pc2.create_cloud_xyz32(b.header, scandots_map)
                self.pub.publish(scandot_msg)
                print("scandot hz : ", 1/(time.time() - prev))
            except:
                print("error")
        else:
            scandots = []
            for i in range(187):
                scandots.append(-0.8577)
                # scandots.append(-1.7577)
            a = Float32MultiArray()
            a.data = scandots
            self.pub_scandot_value.publish(a)
            
            
            
    def ros_to_pcl(self, ros_cloud):
        points_list = []

        for data in pc2.read_points(ros_cloud, skip_nans=True):
            points_list.append([data[0], data[1], data[2], 0.0])
        
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
        self.raw_map = input_ros_msg

if __name__ == "__main__":
    rospy.init_node('scandot', anonymous=True)

    a = scandot_manager()

    rospy.spin()