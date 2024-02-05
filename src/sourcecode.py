import rospy
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np

def inverse_transform_pointcloud(msg, transform_matrix):
    """
    입력된 PointCloud2 메시지의 좌표를 주어진 역변환 행렬로 변환하는 함수

    Parameters:
        msg (PointCloud2): PointCloud2 메시지
        transform_matrix (numpy.ndarray): (4, 4) 크기의 역변환 행렬

    Returns:
        PointCloud2: 역변환된 PointCloud2 메시지
    """
    # 메시지를 numpy 배열로 변환
    points = np.array(list(pc2.read_points(msg)))

    # 변환 행렬을 4x4로 변경 (원점 변환을 위해 homogeneous 좌표로 변환되었을 수 있음)
    if transform_matrix.shape != (4, 4):
        raise ValueError("Invalid transform matrix. Expected shape: (4, 4)")

    # 역변환 적용
    transformed_points = np.dot(np.hstack((points[:, :3], np.ones((len(points), 1)))), transform_matrix.T)
    transformed_points = transformed_points[:, :3]

    # 역변환된 좌표를 PointCloud2 메시지로 변환
    transformed_msg = PointCloud2()
    transformed_msg.header = msg.header
    transformed_msg.height = 1
    transformed_msg.width = len(transformed_points)
    transformed_msg.is_dense = False
    transformed_msg.is_bigendian = False

    fields = [pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
              pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
              pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1)]
    transformed_msg.fields = fields

    transformed_msg.point_step = 12
    transformed_msg.row_step = 12 * len(transformed_points)
    transformed_msg.data = np.asarray(transformed_points, np.float32).tostring()

    return transformed_msg

def pointcloud_callback(msg):
    try:
        # 변환 행렬의 시간 정보를 현재 시간으로 갱신
        transform_matrix_stamped = tf_buffer.lookup_transform("target_frame", msg.header.frame_id, rospy.Time(0))

        # 역변환 적용하여 PointCloud2 메시지 생성
        transformed_msg = inverse_transform_pointcloud(msg, tf2_geometry_msgs.transform_to_matrix(transform_matrix_stamped.transform))

        # 변환된 PointCloud2 메시지를 퍼블리시
        pub.publish(transformed_msg)

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logwarn("Failed to transform PointCloud2 message.")

if __name__ == '__main__':
    # ROS 노드 초기화
    rospy.init_node('pointcloud_inverse_transform_node', anonymous=True)

    # tf2 변환 관리자 생성
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    # PointCloud2 메시지를 받을 토픽에 대한 Subscriber 생성
    rospy.Subscriber('pointcloud_topic', PointCloud2, pointcloud_callback)

    # 변환된 PointCloud2 메시지를 퍼블리시할 Publisher 생성
    pub = rospy.Publisher('transformed_pointcloud_topic', PointCloud2, queue_size=10)

    rospy.spin()