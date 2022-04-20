#/usr/env/bin python
import rospy
from geometry_msgs.msg import PoseStamped
import geometry_msgs
import tf2_ros

def odomCallback(data):
    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = data.header.stamp
    t.header.seq = data.header.seq
    t.header.frame_id = "odom"
    t.child_frame_id = "caddy"
    t.transform.translation.x = data.pose.position.x
    t.transform.translation.y = data.pose.position.y
    t.transform.translation.z = data.pose.position.z + 150.2735
    t.transform.rotation.x = data.pose.orientation.x
    t.transform.rotation.y = data.pose.orientation.y
    t.transform.rotation.z = data.pose.orientation.z
    t.transform.rotation.w = data.pose.orientation.w
    br.sendTransform(t)
    
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = data.header.stamp
    t.header.seq = data.header.seq
    t.header.frame_id = "caddy"
    t.child_frame_id = "camera"
    t.transform.rotation.x = 0
    t.transform.rotation.y = 0.7071068
    t.transform.rotation.z = 0
    t.transform.rotation.w = 0.7071068
    br.sendTransform(t)

def node():
    rospy.init_node("odom_to_map", anonymous=False)
    rospy.Subscriber("/relative/odom", PoseStamped, odomCallback)
    rospy.spin()

if __name__ == "__main__":
    try:
        node()
    except Exception as ex:
        rospy.logerr(ex)
    finally:
        rospy.loginfo("exiting node")