#ifndef __MONOCULAR_INERTIAL_SLAM_NODE_HPP__
#define __MONOCULAR_INERTIAL_SLAM_NODE_HPP__

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/pointcloud2.hpp"
#include "sensor_msgs/msg/pointcloud.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "geometry_msgs/msg/transformstamped.hpp"

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

#TODO: Integrar Tf2 pra publicar tfs e receber tfs pelos tf2_transform_publisher, tf2_static_transform_publisher, etc.
#Inclusive pra que seja melhor integrar todas as transforms pra suprir as transforms necessárias pelo nav2 em outro módulo.

class MonocularInertialSlamNode : public rclcpp::Node
{
public:
    MonocularInertialSlamNode(ORB_SLAM3::System* pSLAM);

    ~MonocularInertialSlamNode();

private:

    void GrabImage(const ImageMsg::SharedPtr msg);
    void GrabImu(const ImuMsg::SharedPtr msg);
    cv::Mat GetImage(const ImageMsg::SharedPtr msg);
    Sophus::SE3f SyncWithImu_Track(); //retorna o resultado do tracking do orbslam3, após sincronizar a imagem e o imu
    TfMsg MakeTfMsg(const Sophus::SE3f Tcm); //inverte transformação e monta mensagem de tf para publição
    void timer_callback();


    ORB_SLAM3::System* m_SLAM;
    std::thread *syncThread_; //inicia thread para sincronizar

    //IMU buffer pointer
    queue<ImuMsg::SharedPtr> imuBuf_;
    std::mutex bufImuMutex_;

    //Image buffer pointer
    queue<ImageMsg::SharedPtr> imgBuf_; 
    std::mutex bufImgMutex_;

    //Transform buffer pointer
    queue<TfMsg::SharedPtr> tfBuf_;
    std::mutex bufTfMutex_;

    //Define subscribers:
    rclcpp::Subscription<ImageMsg>::SharedPtr m_image_subscriber;
    rclcpp::Subscription<ImuMsg>::SharedPtr imu_subscriber;

    //Define publishers:
    rclcpp::Publisher<TfMsg>::SharedPtr twc_publisher; //publisher pra publicar a transformação map => camera (fixa em relação ao base_link)
    rclcpp::Publisher<PclMsg>::SharedPtr pointcloud_publisher; //publisher para publicar pointclouds derivadas dos features detectados TODO: É informação útil?

    //Não entendi bem o que são esses bools:
    bool doRectify_;
    bool doEqual_;
    cv::Mat M1l_, M2l_, M1r_, M2r_;

    bool bClahe_;
    cv::Ptr<cv::CLAHE> clahe_ = cv::createCLAHE(3.0, cv::Size(8, 8));
};

#endif
