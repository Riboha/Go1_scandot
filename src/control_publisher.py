import rospy
from unitree_legged_msgs.msg import Control_12

class control_publisher:

    def __init__(self):

        self.pub = rospy.Publisher("/reinforce_control", Control_12, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(nsecs=20000000), self.control)
        self.iter = 0.0
        self.i = 0
        
    def control(self, timer):
        
        # joints =    [0.0, 0.67, -1.3, 
        #              -0.0, 0.67, -1.3, 
        #             0.0, 0.67, -1.3, 
        #             -0.0, 0.67, -1.3]
        
        joints =    [0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0]
        
        msg = Control_12()
        msg.coltrol_12 = joints
        self.pub.publish(msg)
        if self.i == 0:
            self.iter += 0.01
        else:
            self.iter -= 0.01
            
        if self.iter > 2.0:
            self.i=1
        if self.iter < 0.0:
            self.i=0

if __name__ == "__main__":
    rospy.init_node('tutorial', anonymous=True)

    a = control_publisher()

    rospy.spin()