#include "mono-inertial-node.hpp"
#include <opencv2/core/core.hpp>

using std::placeholders::_1;

MonoInertialNode::MonoInertialNode(ORB_SLAM3::System* pSLAM)
:   Node("ORB_SLAM3_ROS2")
{
    m_SLAM = pSLAM;
    // std::cout << "slam changed" << std::endl;
    image_sub_topic_ = this->declare_parameter<std::string>("image_sub_topic", "/image_raw");
    imu_sub_topic_ = this->declare_parameter<std::string>("imu_sub_topic", "/imu");

    m_image_subscriber = this->create_subscription<ImageMsg>(
        image_sub_topic_,
        rclcpp::SensorDataQoS(),
        std::bind(&MonoInertialNode::GrabImage, this, std::placeholders::_1));
    
    imu_subscriber = this->create_subscription<ImuMsg>(
        imu_sub_topic_,
        rclcpp::SensorDataQoS(),
        std::bind(&MonoInertialNode::GrabImu, this, std::placeholders::_1));

    pointcloud_publisher = this->create_publisher<PclMsg>(
        "PCLTOPIC",
        rclcpp::SensorDataQoS()
    );

    syncThread_ = new std::thread(&MonoInertialNode::SyncWithImu_Track, this);

    std::cout << "slam changed" << std::endl;
}

MonoInertialNode::~MonoInertialNode()
{
    //Delete sync thread
    syncThread_->join();
    delete syncThread_;

    // Stop all threads
    m_SLAM->Shutdown();

    // Save camera trajectory
    m_SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}

void MonoInertialNode::GrabImu(const ImuMsg::SharedPtr msg)
{
    //RCLCPP_INFO(this->get_logger(), "GrabImu chamada");
    //msg->header.stamp = this->get_clock()->now();
    bufImuMutex_.lock();
    if (std::isnan(msg->linear_acceleration.x) || std::isnan(msg->linear_acceleration.y) || std::isnan(msg->linear_acceleration.z) ||
        std::isnan(msg->angular_velocity.x)    || std::isnan(msg->angular_velocity.y)    || std::isnan(msg->angular_velocity.z) ||
        std::isinf(msg->linear_acceleration.x) || std::isinf(msg->linear_acceleration.y) || std::isinf(msg->linear_acceleration.z) ||
        std::isinf(msg->angular_velocity.x)    || std::isinf(msg->angular_velocity.y)    || std::isinf(msg->angular_velocity.z))
    {
        bufImuMutex_.unlock();
        RCLCPP_WARN(this->get_logger(), "IMU descartada: valores NaN ou inf");
        return;
    }
    imuBuf_.push(msg);
    bufImuMutex_.unlock();
    //RCLCPP_INFO(this->get_logger(), "Mensagem IMU recebida");
}

void MonoInertialNode::GrabImage(const ImageMsg::SharedPtr msg)
{
    //RCLCPP_INFO(this->get_logger(), "GrabImage chamada");
    //msg->header.stamp = this->get_clock()->now();
    bufImgMutex_.lock();
    imgBuf_.push(msg);
    bufImgMutex_.unlock();
    //RCLCPP_INFO(this->get_logger(), "Mensagem câmera recebida");
}

cv::Mat MonoInertialNode::GetImage(const ImageMsg::SharedPtr msg)
{
    //RCLCPP_INFO(this->get_logger(), "GetImage chamada");
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr m_cvImPtr;

    try
    {
        m_cvImPtr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception & e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }

    if (m_cvImPtr->image.type() == 0)
    {
        return m_cvImPtr->image.clone();
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "GetImage: Error image type");
        return m_cvImPtr->image.clone();
    }
}

void MonoInertialNode::SyncWithImu_Track()
{
    double tLastImg = -1.0;

    while (rclcpp::ok())
    {
        ImageMsg::SharedPtr img;

        //--------------------------
        // 1. GET IMAGE
        //--------------------------
        {
            std::unique_lock<std::mutex> lock(bufImgMutex_);

            if (imgBuf_.empty())
            {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            img = imgBuf_.front();
            imgBuf_.pop();
        }

        const double tImg = Utility::StampToSec(img->header.stamp);

        // Primeiro frame: apenas memoriza o tempo
        if (tLastImg < 0.0)
        {
            tLastImg = tImg;
            continue;
        }

        std::vector<ORB_SLAM3::IMU::Point> imuData;

        {
            std::unique_lock<std::mutex> lock(bufImuMutex_);

            while (imuBuf_.size() >= 2)
            {
                auto imu0 = imuBuf_.front();
                imuBuf_.pop();

                auto imu1 = imuBuf_.front();

                const double t0 = Utility::StampToSec(imu0->header.stamp);
                const double t1 = Utility::StampToSec(imu1->header.stamp);

                // descarta IMUs anteriores ao frame anterior
                if (t1 <= tLastImg)
                    continue;

                // adiciona medida real
                if (t0 > tLastImg && t0 <= tImg)
                {
                    imuData.emplace_back(
                        cv::Point3f(
                            imu0->linear_acceleration.x,
                            imu0->linear_acceleration.y,
                            imu0->linear_acceleration.z),
                        cv::Point3f(
                            imu0->angular_velocity.x,
                            imu0->angular_velocity.y,
                            imu0->angular_velocity.z),
                        t0);
                }

                // intervalo contém o instante do frame?
                if (t0 < tImg && t1 >= tImg)
                {
                    const double alpha = (tImg - t0) / (t1 - t0);

                    cv::Point3f acc0(
                        imu0->linear_acceleration.x,
                        imu0->linear_acceleration.y,
                        imu0->linear_acceleration.z);

                    cv::Point3f acc1(
                        imu1->linear_acceleration.x,
                        imu1->linear_acceleration.y,
                        imu1->linear_acceleration.z);

                    cv::Point3f gyr0(
                        imu0->angular_velocity.x,
                        imu0->angular_velocity.y,
                        imu0->angular_velocity.z);

                    cv::Point3f gyr1(
                        imu1->angular_velocity.x,
                        imu1->angular_velocity.y,
                        imu1->angular_velocity.z);

                    cv::Point3f acc =
                        acc0 * (1.0 - alpha) + acc1 * alpha;

                    cv::Point3f gyr =
                        gyr0 * (1.0 - alpha) + gyr1 * alpha;

                    imuData.emplace_back(acc, gyr, tImg);

                    // mantém imu1 para o próximo frame
                    break;
                }
            }
        }

        if (imuData.size() < 2)
        {
            continue;
        }

        bool ok = true;

        for (size_t i = 1; i < imuData.size(); ++i)
        {
            double dt = imuData[i].t - imuData[i - 1].t;

            if (!std::isfinite(dt) || dt <= 0.0)
            {
                ok = false;
                break;
            }
        }

        if (!ok)
            continue;

        try
        {
            m_SLAM->TrackMonocular(GetImage(img), tImg, imuData);
            tLastImg = tImg;
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "%s", e.what());
        }
    }
}

/*
            /*Sophus::SE3f Tmc = Tcm.inverse(); //Transformação mapa => camera 
            TfMsg transf_msg;
            try {  
                TfMsg odom_to_base_msg = tf_buffer_->lookupTransform("odom", "base_link", tf2::TimePointZero);
                auto translation = odom_to_base_msg.transform.translation;
                auto rotation = odom_to_base_msg.transform.rotation;
                Eigen::Vector3f trans(translation.x, translation.y, translation.z);
                Eigen::Quaternionf rot(rotation.w, rotation.x, rotation.y, rotation.z);
                Sophus::SE3f Tob(rot, trans);
                Sophus::SE3f Tbo = Tob.inverse();
                Sophus::SE3f Tmo = Tmc * Tbo; //Transformação mapa => odometria, que o nav2 requisita (sem passar camera pra base_link)

                Eigen::Quaternionf Tmo_q(Tmo.rotationMatrix());
                transf_msg.transform.translation.x = Tmo.translation().x();
                transf_msg.transform.translation.y = Tmo.translation().y();
                transf_msg.transform.translation.z = Tmo.translation().z();
                transf_msg.transform.rotation.w = Tmo_q.w();
                transf_msg.transform.rotation.x = Tmo_q.x();
                transf_msg.transform.rotation.y = Tmo_q.y();
                transf_msg.transform.rotation.z = Tmo_q.z();

                transf_msg.header.stamp = this->get_clock()->now();
                transf_msg.header.frame_id = "map";
                transf_msg.child_frame_id = "odom";
                tf_broadcaster_->sendTransform(transf_msg); //Publica transformação pelo tf2
            } catch (const tf2::TransformException & ex) {
                RCLCPP_INFO( this->get_logger(), "Could not find odom to base_link transform: %s", ex.what());
                continue;
            }*//*
            } catch (...) {
                RCLCPP_INFO(this->get_logger(), "Algum problema com o tracking");
                continue;
            }*/
            
            /*if (m_SLAM->GetTrackingState() == ORBSLAM::Tracking::OK){

                Sophus::SE3f Tmc = Tcm.inverse(); //Transformação mapa => camera 
                TfMsg transf_msg;
                try {
                    TfMsg odom_to_base_msg = tf_buffer_->lookupTransform("odom", "base_link", tf2::TimePointZero);
                    auto translation = odom_to_base_msg.transform.translation;
                    auto rotation = odom_to_base_msg.transform.rotation;
                    Eigen::Vector3f trans(translation.x, translation.y, translation.z);
                    Eigen::Quaternionf rot(rotation.w, rotation.x, rotation.y, rotation.z);
                    Sophus::SE3f Tob(rot, trans);
                    Sophus::SE3f Tbo = Tob.inverse();
                    Sophus::SE3f Tmo = Tmc * Tbo; //Transformação mapa => odometria, que o nav2 requisita (sem passar camera pra base_link)

                    Eigen::Quaternionf Tmo_q(Tmo.rotationMatrix());
                    transf_msg.transform.translation.x = Tmo.translation().x();
                    transf_msg.transform.translation.y = Tmo.translation().y();
                    transf_msg.transform.translation.z = Tmo.translation().z();
                    transf_msg.transform.rotation.w = Tmo_q.w();
                    transf_msg.transform.rotation.x = Tmo_q.x();
                    transf_msg.transform.rotation.y = Tmo_q.y();
                    transf_msg.transform.rotation.z = Tmo_q.z();

                    transf_msg.header.stamp = this->get_clock()->now();
                    transf_msg.header.frame_id = "map";
                    transf_msg.child_frame_id = "odom";
                    tf_broadcaster_->sendTransform(transf_msg); //Publica transformação pelo tf2
                } catch (const tf2::TransformException & ex) {
                    RCLCPP_INFO( this->get_logger(), "Could not find odom to base_link transform: %s", ex.what());
                    continue;
                }
            } else {
                RCLCPP_INFO(this->get_logger(), "Algo não está OK com o tracking: %d", m_SLAM->GetTrackingState());*/
       /*     }  
    }
}*/