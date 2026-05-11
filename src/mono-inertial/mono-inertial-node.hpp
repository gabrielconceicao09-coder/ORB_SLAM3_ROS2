#ifndef __MONO_INERTIAL_NODE_HPP__
#define __MONO_INERTIAL_NODE_HPP__

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_cloud.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include<tf2/LinearMath/Transform.h>

#include <cv_bridge/cv_bridge.h>

#include "System.h"
#include "Frame.h"
#include "Map.h"
#include "Tracking.h"

#include "utility.hpp"

using ImuMsg = sensor_msgs::msg::Imu;
using ImageMsg = sensor_msgs::msg::Image;
using TfMsg = geometry_msgs::msg::TransformStamped;
using PclMsg = sensor_msgs::msg::PointCloud2;

//TODO Integrar Tf2 pra publicar tfs e receber tfs pelos tf2_transform_publisher, tf2_static_transform_publisher, etc.
//Inclusive pra que seja melhor integrar todas as transforms pra suprir as transforms necessárias pelo nav2 em outro módulo.

class MonoInertialNode : public rclcpp::Node
{
public:
    MonoInertialNode(ORB_SLAM3::System* pSLAM);

    ~MonoInertialNode();

private:

    void GrabImage(const ImageMsg::SharedPtr msg);
    void GrabImu(const ImuMsg::SharedPtr msg);
    cv::Mat GetImage(const ImageMsg::SharedPtr msg);
    void SyncWithImu_Track(); //publica resultado do tracking do orbslam3, após sincronizar a imagem e o imu


    ORB_SLAM3::System* m_SLAM;
    std::thread *syncThread_; //inicia thread para sincronizar

    //IMU buffer pointer
    queue<ImuMsg::SharedPtr> imuBuf_;
    std::mutex bufImuMutex_;

    //Image buffer pointer
    queue<ImageMsg::SharedPtr> imgBuf_; 
    std::mutex bufImgMutex_;

    //Tracked transform buffer pointer and listened transform buffer
    queue<TfMsg::SharedPtr> tfBuf_;
    std::mutex bufTfMutex_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());

    //Define subscribers:
    rclcpp::Subscription<ImageMsg>::SharedPtr m_image_subscriber;
    rclcpp::Subscription<ImuMsg>::SharedPtr imu_subscriber;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);


    //Define publishers:
    rclcpp::Publisher<PclMsg>::SharedPtr pointcloud_publisher; //publisher para publicar pointclouds derivadas dos features detectados TODO: É informação útil?
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    //Não entendi bem o que são esses bools:
    bool doRectify_;
    bool doEqual_;
    cv::Mat M1l_, M2l_, M1r_, M2r_;

    bool bClahe_;
    cv::Ptr<cv::CLAHE> clahe_ = cv::createCLAHE(3.0, cv::Size(8, 8));
};

#endif
