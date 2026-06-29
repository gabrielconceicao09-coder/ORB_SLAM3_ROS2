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

    const double IMU_MARGIN = 0.02;        // overlap leve (20ms)
    const double BOOTSTRAP_TIME = 0.5;     // mínimo de IMU acumulada
    const double MAX_FRAME_GAP = 0.2;      // evita saltos grandes de câmera

    while (rclcpp::ok())
    {
        ImageMsg::SharedPtr img;

        // =========================
        // 1. GET IMAGE
        // =========================
        {
            std::unique_lock<std::mutex> lock(bufImgMutex_);

            if (imgBuf_.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            img = imgBuf_.front();
            imgBuf_.pop();
        }

        double tImg = Utility::StampToSec(img->header.stamp);

        if (tLastImg > 0 && (tImg - tLastImg) > MAX_FRAME_GAP)
        {
            RCLCPP_WARN(this->get_logger(),
                "FRAME GAP grande: %.3f s (possível reset implícito)",
                tImg - tLastImg);
        }

        // =========================
        // 2. COPY IMU WINDOW (SAFE SNAPSHOT)
        // =========================
        std::vector<sensor_msgs::msg::Imu::SharedPtr> imuSnapshot;

        {
            std::unique_lock<std::mutex> lock(bufImuMutex_);

            if (imuBuf_.size() < 5)
                continue;

            double imuStart = Utility::StampToSec(imuBuf_.front()->header.stamp);
            double imuEnd   = Utility::StampToSec(imuBuf_.back()->header.stamp);

            RCLCPP_INFO(this->get_logger(),
                "IMU buffer window: %.6f → %.6f | img=%.6f | size=%zu",
                imuStart, imuEnd, tImg, imuBuf_.size());

            if ((imuEnd - imuStart) < BOOTSTRAP_TIME)
            {
                RCLCPP_WARN(this->get_logger(),
                    "IMU ainda em bootstrap (%.3f s)",
                    imuEnd - imuStart);
                continue;
            }

            // copia janela SEM destruir buffer ainda
            for (auto &imu : imuBuf_)
            {
                double t = Utility::StampToSec(imu->header.stamp);

                if (t < tImg - IMU_MARGIN)
                    continue;

                if (t > tImg + IMU_MARGIN)
                    break;

                imuSnapshot.push_back(imu);
            }
        }

        if (imuSnapshot.size() < 10)
        {
            RCLCPP_WARN(this->get_logger(),
                "IMU insuficiente para frame (%zu samples)",
                imuSnapshot.size());
            continue;
        }

        // =========================
        // 3. BUILD ORB-SLAM3 IMU PACKET
        // =========================
        std::vector<ORB_SLAM3::IMU::Point> imuData;
        imuData.reserve(imuSnapshot.size());

        double tPrev = -1.0;

        for (auto &imu : imuSnapshot)
        {
            double t = Utility::StampToSec(imu->header.stamp);

            if (tPrev > 0)
            {
                double dt = t - tPrev;

                if (!std::isfinite(dt) || dt <= 0.0)
                {
                    RCLCPP_ERROR(this->get_logger(),
                        "DT inválido: %.9f (corrigindo skip)", dt);
                    continue;
                }

                if (dt > 0.05)
                {
                    RCLCPP_WARN(this->get_logger(),
                        "DT grande: %.6f", dt);
                }
            }

            cv::Point3f acc(
                imu->linear_acceleration.x,
                imu->linear_acceleration.y,
                imu->linear_acceleration.z);

            cv::Point3f gyr(
                imu->angular_velocity.x,
                imu->angular_velocity.y,
                imu->angular_velocity.z);

            imuData.emplace_back(acc, gyr, t);

            tPrev = t;
        }

        // =========================
        // 4. CONSUME IMU BUFFER (SAFE ADVANCE)
        // =========================
        {
            std::unique_lock<std::mutex> lock(bufImuMutex_);

            while (!imuBuf_.empty())
            {
                double t = Utility::StampToSec(imuBuf_.front()->header.stamp);

                // só remove até o início da janela
                if (t < tImg - IMU_MARGIN)
                {
                    imuBuf_.pop();
                }
                else
                {
                    break;
                }
            }
        }

        // =========================
        // 5. SLAM CALL
        // =========================
        try
        {
            RCLCPP_INFO(this->get_logger(),
                "TrackMonocular: img=%.6f imu=%zu range=[%.6f → %.6f]",
                tImg,
                imuData.size(),
                imuData.front().t,
                imuData.back().t);

            m_SLAM->TrackMonocular(GetImage(img), tImg, imuData);

            tLastImg = tImg;
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(),
                "SLAM error: %s", e.what());
        }
    }
}

//Versão simples que roda mas perde rápido o tracking:
/*void MonoInertialNode::SyncWithImu_Track()
{
    while (rclcpp::ok())
    {
        ImageMsg::SharedPtr img;

        // =========================
        // 1. GET IMAGE (NO TIME MOD)
        // =========================
        {
            std::unique_lock<std::mutex> lock(bufImgMutex_);

            if (imgBuf_.empty()) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            img = imgBuf_.front();
            imgBuf_.pop();
        }

        double tImg = Utility::StampToSec(img->header.stamp);

        // =========================
        // 2. COLLECT IMU UNTIL IMAGE TIME
        // =========================
        std::vector<ORB_SLAM3::IMU::Point> imuData;

        {
            std::unique_lock<std::mutex> lock(bufImuMutex_);

            if (imuBuf_.size() < 2) {
                continue;
            }

            double lastImgTime = tImg;

            while (!imuBuf_.empty())
            {
                auto imu = imuBuf_.front();
                double t = Utility::StampToSec(imu->header.stamp);

                if (t > lastImgTime)
                    break;

                cv::Point3f acc(
                    imu->linear_acceleration.x,
                    imu->linear_acceleration.y,
                    imu->linear_acceleration.z);

                cv::Point3f gyr(
                    imu->angular_velocity.x,
                    imu->angular_velocity.y,
                    imu->angular_velocity.z);

                imuData.emplace_back(acc, gyr, t);

                imuBuf_.pop();
            }
        }

        // =========================
        // 3. VALIDATE (VERY IMPORTANT)
        // =========================
        if (imuData.size() < 2)
            continue;

        // check dt consistency
        for (size_t i = 1; i < imuData.size(); i++)
        {
            double dt = imuData[i].t - imuData[i - 1].t;

            if (!std::isfinite(dt) || dt <= 0.0)
            {
                RCLCPP_ERROR(this->get_logger(), "BAD IMU DT DETECTED");
                return;
            }
        }

        // =========================
        // 4. RUN SLAM
        // =========================
        try {
            m_SLAM->TrackMonocular(GetImage(img), tImg, imuData);
        }
        catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "SLAM error: %s", e.what());
        }
    }
}*/
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