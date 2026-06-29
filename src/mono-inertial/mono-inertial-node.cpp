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

//Solução Gemini:
void MonoInertialNode::SyncWithImu_Track()
{
    // Armazena o timestamp do último frame processado com sucesso
    double lastProcessedImgTime = -1.0;

    while (rclcpp::ok())
    {
        ImageMsg::SharedPtr imgMsg;

        // =================================================================
        // 1. OBTENÇÃO DA PRÓXIMA IMAGEM DO BUFFER (SEM MODIFICAÇÃO DE TEMPO)
        // =================================================================
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

        // =================================================================
        // 2. VERIFICAÇÃO DE DISPONIBILIDADE DA IMU (AGUARDANDO DADOS FUTUROS)
        // =================================================================
        {
            std::unique_lock<std::mutex> lock(bufImuMutex_);

            // Se o buffer estiver vazio ou o último dado da IMU for anterior à imagem,
            // precisamos esperar a IMU chegar para garantir que cobrimos o tempo da imagem.
            if (imuBuf_.empty() || Utility::StampToSec(imuBuf_.back()->header.stamp) < tImg) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
        }

        // Se chegamos aqui, podemos remover a imagem do buffer com segurança
        {
            std::unique_lock<std::mutex> lock(bufImgMutex_);
            imgBuf_.pop();
        }

        // =================================================================
        // 3. COLETA E FILTRAGEM DOS DADOS DA IMU PARA ESTA IMAGEM
        // =================================================================
        std::vector<ORB_SLAM3::IMU::Point> imuData;

        {
            std::unique_lock<std::mutex> lock(bufImuMutex_);

            while (!imuBuf_.empty())
            {
                auto imuMsg = imuBuf_.front();
                double tImu = Utility::StampToSec(imuMsg->header.stamp);

                // Coleta todos os dados até o tempo da imagem atual (e o primeiro ponto imediatamente após)
                if (tImu > tImg) {
                    // Inclui este ponto que passou ligeiramente do tempo da imagem (essencial para interpolação)
                    cv::Point3f acc(imuMsg->linear_acceleration.x, imuMsg->linear_acceleration.y, imuMsg->linear_acceleration.z);
                    cv::Point3f gyr(imuMsg->angular_velocity.x, imuMsg->angular_velocity.y, imuMsg->angular_velocity.z);
                    imuData.emplace_back(acc, gyr, tImu);
                    break;
                }

                // Processa pontos válidos dentro do intervalo correto
                cv::Point3f acc(imuMsg->linear_acceleration.x, imuMsg->linear_acceleration.y, imuMsg->linear_acceleration.z);
                cv::Point3f gyr(imuMsg->angular_velocity.x, imuMsg->angular_velocity.y, imuMsg->angular_velocity.z);
                imuData.emplace_back(acc, gyr, tImu);

                imuBuf_.pop();
            }
        }

        // =================================================================
        // 4. VALIDAÇÃO CONSISTENTE DOS DADOS COLETADOS
        // =================================================================
        // O ORB-SLAM3 precisa de pelo menos 1 ponto (idealmente múltiplos) para integrar
        if (imuData.empty()) {
            RCLCPP_WARN(this->get_logger(), "Nenhum dado de IMU disponivel para o frame no tempo: %f", tImg);
            continue;
        }

        // Validação de monotonicidade e integridade do dt (Delta Time)
        bool bad_imu = false;
        for (size_t i = 1; i < imuData.size(); i++)
        {
            double dt = imuData[i].t - imuData[i - 1].t;
            if (!std::isfinite(dt) || dt <= 0.0) {
                RCLCPP_ERROR(this->get_logger(), "DT INVALIDO OU NEGATIVO DETECTADO NA IMU: %f", dt);
                bad_imu = true;
                break;
            }
        }

        if (bad_imu) {
            continue; // Descarta o frame se os dados de tempo da IMU vierem corrompidos
        }

        // Guardrail: Evita enviar imagens fora de ordem temporal para o ORB-SLAM3
        if (lastProcessedImgTime > 0.0 && tImg <= lastProcessedImgTime) {
            RCLCPP_WARN(this->get_logger(), "Imagem rejeitada: Timestamp antigo detectado (%f <= %f)", tImg, lastProcessedImgTime);
            continue;
        }

        // =================================================================
        // 5. CONVERSÃO DA IMAGEM E EXECUÇÃO DO TRACKING DO SLAM
        // =================================================================
        cv::Mat cvImage = GetImage(imgMsg);
        if (cvImage.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Falha ao converter a imagem para cv::Mat");
            continue;
        }

        try 
        {
            // Executa o pipeline de rastreamento Mono-Inercial do ORB-SLAM3
            m_SLAM->TrackMonocular(cvImage, tImg, imuData);
            lastProcessedImgTime = tImg;
        }
        catch (const std::exception &e) 
        {
            RCLCPP_ERROR(this->get_logger(), "Erro critico no pipeline do ORB-SLAM3: %s", e.what());
        }
    }
}


/*//Solução ChatGPT4
void MonoInertialNode::SyncWithImu_Track()
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