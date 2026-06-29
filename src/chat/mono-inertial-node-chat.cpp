#include "mono-inertial-chat.hpp"

#include <chrono>

using namespace std::chrono_literals;

//
// ============================================================================
// Constructor
// ============================================================================
//

MonoInertialNode::MonoInertialNode()
    : Node("mono_inertial_chat"),
      running_(true),
      last_image_time_(-1.0)
{
    //--------------------------------------------------------
    // Parameters
    //--------------------------------------------------------

    vocabulary_file_ =
        this->declare_parameter<std::string>(
            "voc_file", "");

    settings_file_ =
        this->declare_parameter<std::string>(
            "settings_file", "");

    image_topic_ =
        this->declare_parameter<std::string>(
            "image_topic",
            "/image_raw");

    imu_topic_ =
        this->declare_parameter<std::string>(
            "imu_topic",
            "/imu");

    use_viewer_ =
        this->declare_parameter<bool>(
            "viewer",
            true);

    verbose_ =
        this->declare_parameter<bool>(
            "verbose",
            false);

    imu_overlap_ =
        this->declare_parameter<double>(
            "imu_overlap",
            0.005);

    imu_bootstrap_time_ =
        this->declare_parameter<double>(
            "imu_bootstrap_time",
            0.50);

    imu_time_shift_ =
        this->declare_parameter<double>(
            "imu_time_shift",
            0.0);

    //--------------------------------------------------------
    // ORB-SLAM3
    //--------------------------------------------------------

    RCLCPP_INFO(
        this->get_logger(),
        "Creating ORB-SLAM3 System...");

    slam_ = std::make_unique<ORB_SLAM3::System>(
        vocabulary_file_,
        settings_file_,
        ORB_SLAM3::System::IMU_MONOCULAR,
        use_viewer_);

    RCLCPP_INFO(
        this->get_logger(),
        "ORB-SLAM3 created.");

    //--------------------------------------------------------
    // Subscribers
    //--------------------------------------------------------

    image_sub_ =
        this->create_subscription<ImageMsg>(
            image_topic_,
            rclcpp::SensorDataQoS(),
            std::bind(
                &MonoInertialNode::ImageCallback,
                this,
                std::placeholders::_1));

    imu_sub_ =
        this->create_subscription<ImuMsg>(
            imu_topic_,
            rclcpp::SensorDataQoS(),
            std::bind(
                &MonoInertialNode::ImuCallback,
                this,
                std::placeholders::_1));

    RCLCPP_INFO(
        this->get_logger(),
        "Subscribers created.");

    //--------------------------------------------------------
    // Sync Thread
    //--------------------------------------------------------

    sync_thread_ =
        std::thread(
            &MonoInertialNode::SyncWithImu_Track,
            this);

    RCLCPP_INFO(
        this->get_logger(),
        "Synchronization thread started.");
}

//
// ============================================================================
// Destructor
// ============================================================================
//

MonoInertialNode::~MonoInertialNode()
{
    running_ = false;

    if (sync_thread_.joinable())
        sync_thread_.join();

    if (slam_)
    {
        RCLCPP_INFO(
            this->get_logger(),
            "Shutting down ORB-SLAM3...");

        slam_->Shutdown();

        RCLCPP_INFO(
            this->get_logger(),
            "ORB-SLAM3 shutdown complete.");
    }
}

//
// ============================================================================
// Image callback
// ============================================================================
//

void MonoInertialNode::ImageCallback(
    const ImageMsg::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(image_mutex_);

    image_buffer_.push_back(msg);

    if (verbose_)
    {
        RCLCPP_INFO(
            this->get_logger(),
            "Image received | buffer=%zu",
            image_buffer_.size());
    }
}

//
// ============================================================================
// IMU callback
// ============================================================================
//

void MonoInertialNode::ImuCallback(
    const ImuMsg::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(imu_mutex_);

    imu_buffer_.push_back(msg);

    //--------------------------------------------------------
    // Prevent unlimited growth
    //--------------------------------------------------------

    constexpr size_t MAX_IMU_BUFFER = 10000;

    while (imu_buffer_.size() > MAX_IMU_BUFFER)
        imu_buffer_.pop_front();

    if (verbose_)
    {
        RCLCPP_INFO(
            this->get_logger(),
            "IMU received | buffer=%zu",
            imu_buffer_.size());
    }
}

//
// ============================================================================
// Convert Image
// ============================================================================
//

cv::Mat MonoInertialNode::GetImage(
    const ImageMsg::SharedPtr msg) const
{
    cv_bridge::CvImageConstPtr cv_ptr;

    try
    {
        cv_ptr =
            cv_bridge::toCvShare(
                msg,
                msg->encoding);
    }
    catch (cv_bridge::Exception &e)
    {
        throw std::runtime_error(
            std::string("cv_bridge exception: ") +
            e.what());
    }

    return cv_ptr->image.clone();
}

//
// ============================================================================
// Timestamp helper
// ============================================================================
//

double MonoInertialNode::StampToSec(
    const builtin_interfaces::msg::Time &stamp) const
{
    return static_cast<double>(stamp.sec) +
           static_cast<double>(stamp.nanosec) * 1e-9;
}

void MonoInertialNode::SyncWithImu_Track()
{
    while (running_ && rclcpp::ok())
    {
        //----------------------------------------------------
        // Wait image
        //----------------------------------------------------

        ImageMsg::SharedPtr image;

        {
            std::lock_guard<std::mutex> lock(image_mutex_);

            if (image_buffer_.empty())
            {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(2));
                continue;
            }

            image = image_buffer_.front();
        }

        const double tImage =
            StampToSec(image->header.stamp) + imu_time_shift_;

        //----------------------------------------------------
        // Copy IMU packet
        //----------------------------------------------------

        std::vector<ORB_SLAM3::IMU::Point> imuPacket;

        {
            std::lock_guard<std::mutex> lock(imu_mutex_);

            if (imu_buffer_.size() < 2)
            {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(2));
                continue;
            }

            double imuBegin =
                StampToSec(imu_buffer_.front()->header.stamp);

            double imuEnd =
                StampToSec(imu_buffer_.back()->header.stamp);

            //------------------------------------------------
            // Bootstrap
            //------------------------------------------------

            if ((imuEnd - imuBegin) < imu_bootstrap_time_)
            {
                continue;
            }

            //------------------------------------------------
            // Wait future imu
            //------------------------------------------------

            if (imuEnd < tImage)
            {
                continue;
            }

            //------------------------------------------------
            // Build packet
            //------------------------------------------------

            double startTime =
                (last_image_time_ < 0.0)
                    ? imuBegin
                    : last_image_time_ - imu_overlap_;

            double previous = -1.0;

            for (const auto &imu : imu_buffer_)
            {
                double t =
                    StampToSec(
                        imu->header.stamp);

                if (t < startTime)
                    continue;

                if (t > tImage)
                    break;

                if (previous > 0.0)
                {
                    double dt = t - previous;

                    if (!std::isfinite(dt) ||
                        dt <= 0.0)
                    {
                        RCLCPP_ERROR(
                            get_logger(),
                            "Invalid IMU dt");

                        imuPacket.clear();
                        break;
                    }
                }

                previous = t;

                cv::Point3f acc(
                    imu->linear_acceleration.x,
                    imu->linear_acceleration.y,
                    imu->linear_acceleration.z);

                cv::Point3f gyr(
                    imu->angular_velocity.x,
                    imu->angular_velocity.y,
                    imu->angular_velocity.z);

                imuPacket.emplace_back(
                    acc,
                    gyr,
                    t);
            }
        }

        //----------------------------------------------------
        // Validation
        //----------------------------------------------------

        if (imuPacket.size() < 2)
        {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(2));
            continue;
        }

        //----------------------------------------------------
        // Convert image
        //----------------------------------------------------

        cv::Mat img;

        try
        {
            img = GetImage(image);
        }
        catch (...)
        {
            continue;
        }

        //----------------------------------------------------
        // Track
        //----------------------------------------------------

        try
        {
            slam_->TrackMonocular(
                img,
                tImage,
                imuPacket);
        }
        catch (...)
        {
            continue;
        }

        //----------------------------------------------------
        // Success
        //----------------------------------------------------

        last_image_time_ = tImage;

        //----------------------------------------------------
        // Remove image
        //----------------------------------------------------

        {
            std::lock_guard<std::mutex> lock(
                image_mutex_);

            if (!image_buffer_.empty() &&
                image_buffer_.front() == image)
            {
                image_buffer_.pop_front();
            }
        }

        //----------------------------------------------------
        // Prune IMU
        //----------------------------------------------------

        {
            std::lock_guard<std::mutex> lock(
                imu_mutex_);

            double pruneTime =
                last_image_time_ - imu_overlap_;

            while (imu_buffer_.size() > 2)
            {
                double t =
                    StampToSec(
                        imu_buffer_.front()->header.stamp);

                if (t >= pruneTime)
                    break;

                imu_buffer_.pop_front();
            }
        }

        //----------------------------------------------------
        // Debug
        //----------------------------------------------------

        if (verbose_)
        {
            RCLCPP_INFO(
                get_logger(),
                "Track OK | image %.6f | imu %zu",
                tImage,
                imuPacket.size());
        }
    }
}