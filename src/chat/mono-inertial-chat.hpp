#ifndef MONO_INERTIAL_CHAT_HPP
#define MONO_INERTIAL_CHAT_HPP

#include <deque>
#include <thread>
#include <mutex>
#include <memory>
#include <atomic>
#include <string>

// ROS2
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <cv_bridge/cv_bridge.h>

// OpenCV
#include <opencv2/opencv.hpp>

// ORB-SLAM3
#include "System.h"

//------------------------------------------------------------
// Aliases
//------------------------------------------------------------

using ImageMsg = sensor_msgs::msg::Image;
using ImuMsg   = sensor_msgs::msg::Imu;

//------------------------------------------------------------
// MonoInertialNode
//------------------------------------------------------------

class MonoInertialNode : public rclcpp::Node
{

public:

    //--------------------------------------------------------
    // Constructor / Destructor
    //--------------------------------------------------------

    MonoInertialNode();

    ~MonoInertialNode();

private:

    //--------------------------------------------------------
    // ROS Callbacks
    //--------------------------------------------------------

    void ImageCallback(
        const ImageMsg::SharedPtr msg);

    void ImuCallback(
        const ImuMsg::SharedPtr msg);

    //--------------------------------------------------------
    // Synchronization thread
    //--------------------------------------------------------

    void SyncWithImu_Track();

    //--------------------------------------------------------
    // Utilities
    //--------------------------------------------------------

    cv::Mat GetImage(
        const ImageMsg::SharedPtr msg) const;

    double StampToSec(
        const builtin_interfaces::msg::Time &stamp) const;

    //--------------------------------------------------------
    // Parameters
    //--------------------------------------------------------

    std::string vocabulary_file_;

    std::string settings_file_;

    std::string image_topic_;

    std::string imu_topic_;

    bool use_viewer_;

    double imu_overlap_;

    double imu_bootstrap_time_;

    double imu_time_shift_;

    //--------------------------------------------------------
    // ORB-SLAM3
    //--------------------------------------------------------

    std::unique_ptr<ORB_SLAM3::System> slam_;

    //--------------------------------------------------------
    // Subscribers
    //--------------------------------------------------------

    rclcpp::Subscription<ImageMsg>::SharedPtr image_sub_;

    rclcpp::Subscription<ImuMsg>::SharedPtr imu_sub_;

    //--------------------------------------------------------
    // Buffers
    //--------------------------------------------------------

    std::deque<ImageMsg::SharedPtr> image_buffer_;

    std::deque<ImuMsg::SharedPtr> imu_buffer_;

    //--------------------------------------------------------
    // Mutexes
    //--------------------------------------------------------

    std::mutex image_mutex_;

    std::mutex imu_mutex_;

    //--------------------------------------------------------
    // Synchronization Thread
    //--------------------------------------------------------

    std::thread sync_thread_;

    std::atomic<bool> running_;

    //--------------------------------------------------------
    // Timing
    //--------------------------------------------------------

    double last_image_time_;

    //--------------------------------------------------------
    // Debug
    //--------------------------------------------------------

    bool verbose_;

};

#endif