using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using rclcs;
using System;

public class CameraPublisher : MonoBehaviourRosNode
{
    public string NodeName = "camera_node";
    protected override string nodeName { get { return NodeName; } }
    public string CamTopic = "pi_camera/compressed";
    public string CamInfoTopic = "pi_camera/camera_info";
    public string ImageTopic = "pi_camera/image";
    private Publisher<sensor_msgs.msg.CompressedImage> camPublisher;
    private Publisher<sensor_msgs.msg.CameraInfo> camInfoPublisher;
    private Publisher<sensor_msgs.msg.Image> imagePublisher;
    public Camera ImageCamera;
    public string FrameId = "/base_footprint";
    public int resolutionWidth = 640;
    public int resolutionHeight = 480;
    [Range(0, 100)]
    public int qualityLevel = 50;
    private sensor_msgs.msg.CompressedImage compImageMsg;
    private sensor_msgs.msg.Image imageMsg;
    private sensor_msgs.msg.CameraInfo camera_info;
    private Texture2D texture2D;
    private Rect rect;


    
    protected override void StartRos()
    {
        camPublisher = node.CreatePublisher<sensor_msgs.msg.CompressedImage>(CamTopic);
        camInfoPublisher = node.CreatePublisher<sensor_msgs.msg.CameraInfo>(CamInfoTopic);
        imagePublisher = node.CreatePublisher<sensor_msgs.msg.Image>(ImageTopic);
        InitializeGameObject();
        InitializeMessage();
        setCameraInfo();
        Camera.onPostRender += UpdateImage;
    }
    
    private void setCameraInfo()
    {
        camera_info = new sensor_msgs.msg.CameraInfo();
        camera_info.Header.Frame_id = FrameId;   
    }

    private void Update()
    {
        Camera.onPostRender += UpdateImage;
        updateCameraInfo();
    }

    private void updateCameraInfo()
    {
        camera_info.Header.Frame_id = FrameId;
        camera_info.Header.Update(clock);
        camera_info.Width = Convert.ToUInt32(resolutionWidth);
        camera_info.Height = Convert.ToUInt32(resolutionHeight);
        camInfoPublisher.Publish(camera_info);
    }

    private void UpdateImage(Camera _camera)
    {
        if (texture2D != null && _camera == this.ImageCamera)
        {
            UpdateMessage();
        }
    }

    private void InitializeGameObject()
    {
        texture2D = new Texture2D(resolutionWidth, resolutionHeight, TextureFormat.RGB24, false);
        rect = new Rect(0, 0, resolutionWidth, resolutionHeight);
        ImageCamera.targetTexture = new RenderTexture(resolutionWidth, resolutionHeight, 24);
    }
    private void InitializeMessage()
    {
        compImageMsg = new sensor_msgs.msg.CompressedImage();
        imageMsg = new sensor_msgs.msg.Image();
        compImageMsg.Header.Frame_id = FrameId;
        imageMsg.Header.Frame_id = FrameId;
        compImageMsg.Format = "jpeg";

        imageMsg.Width = Convert.ToUInt32(resolutionWidth);
        imageMsg.Height = Convert.ToUInt32(resolutionHeight);
        imageMsg.Step = Convert.ToUInt32(resolutionWidth*4);
    }
    private void UpdateMessage()
    {
        compImageMsg.Header.Update(clock);
        texture2D.ReadPixels(rect, 0, 0);
        compImageMsg.Data = texture2D.EncodeToJPG(qualityLevel); 
        camPublisher.Publish(compImageMsg);

        imageMsg.Header.Update(clock);
        imageMsg.Data = texture2D.EncodeToJPG(qualityLevel);
        imageMsg.Encoding = "rgb8";
        imageMsg.Step = Convert.ToUInt32(resolutionWidth*4);
        imagePublisher.Publish(imageMsg);

    }
}
