#/usr/env/bin python
import rospy
from nav_msgs.msg import Odometry
import geometry_msgs
import tf2_ros
global start_z
global first_data
first_data = True
start_z = 0

def odomCallback(data):
    global first_data
    global start_z

    if first_data:
        first_data = False
        start_z = data.pose.pose.position.z
    else:
        br = tf2_ros.TransformBroadcaster()
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x = data.pose.pose.position.x
        t.transform.translation.y = data.pose.pose.position.y
        t.transform.translation.z = data.pose.pose.position.z - start_z
        t.transform.rotation.x = data.pose.pose.orientation.x
        t.transform.rotation.y = data.pose.pose.orientation.y
        t.transform.rotation.z = data.pose.pose.orientation.z
        t.transform.rotation.w = data.pose.pose.orientation.w
        
        br.sendTransform(t)
        rospy.loginfo("tf")

def node():
    rospy.init_node("odom_to_tf", anonymous=False)
    rospy.Subscriber("odom", Odometry, odomCallback)
    rospy.spin()

if __name__ == "__main__":
    try:
        node()
    except Exception as ex:
        rospy.logerr(ex)
    finally:
        rospy.loginfo("exiting node")