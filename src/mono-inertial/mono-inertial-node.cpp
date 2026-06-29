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
    double lastProcessedImgTime = -1.0;

    while (rclcpp::ok())
    {
        ImageMsg::SharedPtr img;

        // ==========================================
        // 1. ESPERA E OBTÉM A IMAGEM (SEM POP AINDA)
        // ==========================================
        {
            std::unique_lock<std::mutex> lock(bufImgMutex_);

            if (imgBuf_.empty()) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            img = imgBuf_.front();
        }

        double tImg = Utility::StampToSec(img->header.stamp);

        // ==========================================
        // 2. VERIFICA SE A IMU JÁ ALCANÇOU A IMAGEM
        // ==========================================
        {
            std::unique_lock<std::mutex> lock(bufImuMutex_);

            // Se o buffer da IMU estiver muito vazio ou o último dado da IMU 
            // for mais velho que a imagem, esperamos a IMU chegar.
            if (imuBuf_.empty() || Utility::StampToSec(imuBuf_.back()->header.stamp) < tImg) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue; 
            }
        }

        // Agora que temos certeza que a IMU cobriu o tempo da imagem, 
        // removemos a imagem do buffer com segurança.
        {
            std::unique_lock<std::mutex> lock(bufImgMutex_);
            imgBuf_.pop();
        }

        // ==========================================
        // 3. COLA OS DADOS DA IMU ATÉ O INSTANTE DA IMAGEM
        // ==========================================
        std::vector<ORB_SLAM3::IMU::Point> imuData;

        {
            std::unique_lock<std::mutex> lock(bufImuMutex_);

            while (!imuBuf_.empty())
            {
                auto imu = imuBuf_.front();
                double t = Utility::StampToSec(imu->header.stamp);

                cv::Point3f acc(imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z);
                cv::Point3f gyr(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);

                // IMPORTANTE: Adiciona o ponto atual no vetor
                imuData.emplace_back(acc, gyr, t);
                imuBuf_.pop();

                // Regra do ORB-SLAM3: Precisamos de pelo menos UM ponto cujo 
                // tempo seja maior (futuro) que o tempo da imagem para interpolar.
                if (t > tImg) {
                    break; 
                }
            }
        }

        // ==========================================
        // 4. VALIDAÇÃO E FILTRAGEM DE REPETIÇÃO
        // ==========================================
        if (imuData.size() < 2) {
            continue;
        }

        // Garante monotonicidade temporal estrita para o ORB-SLAM3
        if (lastProcessedImgTime > 0.0 && tImg <= lastProcessedImgTime) {
            continue;
        }

        // Verifica consistência do DT
        bool bad_dt = false;
        for (size_t i = 1; i < imuData.size(); i++) {
            double dt = imuData[i].t - imuData[i - 1].t;
            if (!std::isfinite(dt) || dt <= 0.0) {
                RCLCPP_ERROR(this->get_logger(), "BAD IMU DT DETECTED: %f", dt);
                bad_dt = true;
                break;
            }
        }
        if (bad_dt) continue;

        // ==========================================
        // 5. ENVIA PARA O SLAM
        // ==========================================
        cv::Mat cvImage = GetImage(img);
        if (cvImage.empty()) {
            continue;
        }

        try {
            m_SLAM->TrackMonocular(cvImage, tImg, imuData);
            lastProcessedImgTime = tImg;
        }
        catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "SLAM error: %s", e.what());
        }
    }
}

/*//Solução Gemini:
void MonoInertialNode::SyncWithImu_Track()
{
    double lastProcessedImgTime = -1.0;
    int warning_counter = 0;

    while (rclcpp::ok())
    {
        ImageMsg::SharedPtr imgMsg;

        // 1. GET IMAGE
        {
            std::unique_lock<std::mutex> lock(bufImgMutex_);
            if (imgBuf_.empty()) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            imgMsg = imgBuf_.front();
        }

        double tImg = Utility::StampToSec(imgMsg->header.stamp);

        // 2. CHECK IMU BUFFER AND DIAGNOSTIC
        {
            std::unique_lock<std::mutex> lock(bufImuMutex_);

            if (imuBuf_.empty()) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            double tLastImu = Utility::StampToSec(imuBuf_.back()->header.stamp);
            double tFirstImu = Utility::StampToSec(imuBuf_.front()->header.stamp);

            // Se o último dado da IMU ainda é mais velho que a imagem, precisamos esperar.
            // Mas se a IMU estiver MUITO atrás (ex: mais de 1 segundo), há um erro de sincronização geral.
            if (tLastImu < tImg) {
                warning_counter++;
                if (warning_counter % 100 == 0) {
                    RCLCPP_WARN(this->get_logger(), 
                        "IMU esta ATRASADA em relacao a Imagem! [tImg: %.4f | Ultima IMU: %.4f | Diferenca: %.4f s]", 
                        tImg, tLastImu, tImg - tLastImu);
                }
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
        }

        // Remove a imagem do buffer pois a IMU já a alcançou temporalmente
        {
            std::unique_lock<std::mutex> lock(bufImgMutex_);
            imgBuf_.pop();
        }

        // 3. COLLECT IMU DATA
        std::vector<ORB_SLAM3::IMU::Point> imuData;

        {
            std::unique_lock<std::mutex> lock(bufImuMutex_);

            while (!imuBuf_.empty())
            {
                auto imuMsg = imuBuf_.front();
                double tImu = Utility::StampToSec(imuMsg->header.stamp);

                cv::Point3f acc(imuMsg->linear_acceleration.x, imuMsg->linear_acceleration.y, imuMsg->linear_acceleration.z);
                cv::Point3f gyr(imuMsg->angular_velocity.x, imuMsg->angular_velocity.y, imuMsg->angular_velocity.z);
                
                // IMPORTANTE: Adiciona o ponto antes de checar o limite superior
                imuData.emplace_back(acc, gyr, tImu);
                imuBuf_.pop();

                // Se já passamos do tempo da imagem, coletamos o suficiente para este frame
                if (tImu > tImg) {
                    break;
                }
            }
        }

        // 4. VALIDATE
        if (imuData.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Vetor de IMU vazio gerado no wrapper!");
            continue;
        }

        // Evita que o ORB-SLAM3 estoure por falta de monotonicidade temporal das imagens
        if (lastProcessedImgTime > 0.0 && tImg <= lastProcessedImgTime) {
            continue;
        }

        // 5. RUN SLAM
        cv::Mat cvImage = GetImage(imgMsg);
        if (cvImage.empty()) {
            continue;
        }

        try {
            m_SLAM->TrackMonocular(cvImage, tImg, imuData);
            lastProcessedImgTime = tImg;
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