using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.UrdfImporter;
using System;
using rclcs;

public class JointStatePublisher : MonoBehaviourRosNode
{
    public GameObject left_wheel;
    public GameObject right_wheel;
    public string NodeName = "joint_states_node";
    protected override string nodeName { get { return NodeName; } }
    public string topicName = "joint_states";
    public string frame_id = "";
    public float ScanningFrequency = 10f;
    private sensor_msgs.msg.JointState joint_msgs;
    private Publisher<sensor_msgs.msg.JointState> jointStatePub;
    public string[] jointForJointState = { "wheel_right_joint", "wheel_left_joint" };


    // Start is called before the first frame update

    protected override void StartRos()
    {
        init_msgs();
        jointStatePub = node.CreatePublisher<sensor_msgs.msg.JointState>(topicName);

        //StartCoroutine("JointPublish");
    }

    private void init_msgs()
    {
        joint_msgs = new sensor_msgs.msg.JointState();
        int len_joint = jointForJointState.Length;
        joint_msgs.Header.Frame_id = frame_id;
        joint_msgs.Name = new string[len_joint];
        joint_msgs.Position = new double[len_joint];
        joint_msgs.Velocity = new double[len_joint];
        joint_msgs.Effort = new double[len_joint];
    }

    // Update is called once per frame
    void Update()
    {

        joint_msgs.Header.Update(clock);
        joint_msgs.Position[0] = right_wheel.transform.localRotation.eulerAngles.x * Mathf.Deg2Rad;
        joint_msgs.Position[1] = left_wheel.transform.localRotation.eulerAngles.x * Mathf.Deg2Rad;
        joint_msgs.Velocity[0] = 0;
        joint_msgs.Velocity[1] = 0;
        joint_msgs.Effort[0] = 0;
        joint_msgs.Effort[1] = 0;

        joint_msgs.Name = jointForJointState;
        jointStatePub.Publish(joint_msgs);
    }
    
}