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
        std::isnan(msg->angular_velocity.x)    || std::isnan(msg->angular_velocity.y)    || std::isnan(msg->angular_velocity.z))
    {
        bufImuMutex_.unlock();
        RCLCPP_WARN(this->get_logger(), "IMU descartada: Contém valores NaN!");
        return;
    }
    imuBuf_.push(msg);
    bufImuMutex_.unlock();
    RCLCPP_INFO(this->get_logger(), "Mensagem IMU recebida");
}

void MonoInertialNode::GrabImage(const ImageMsg::SharedPtr msg)
{
    //RCLCPP_INFO(this->get_logger(), "GrabImage chamada");
    //msg->header.stamp = this->get_clock()->now();
    bufImgMutex_.lock();
    imgBuf_.push(msg);
    bufImgMutex_.unlock();
    RCLCPP_INFO(this->get_logger(), "Mensagem câmera recebida");
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
        RCLCPP_INFO(this->get_logger(), "GetImage rodou");
        return m_cvImPtr->image.clone();
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Error image type");
        RCLCPP_INFO(this->get_logger(), "GetImage rodou");
        return m_cvImPtr->image.clone();
    }
}


void MonoInertialNode::SyncWithImu_Track()
{
    double tLastImg = -1.0;
    const double kalibr_timeshift = -0.0185970677;

    while (rclcpp::ok())
    {
        cv::Mat Img;
        ImageMsg::SharedPtr img_msg = nullptr;

        double tImg_cam = 0.0;
        double tImg = 0.0;

        // =========================================================
        // 1. GET IMAGE (NON-BLOCKING)
        // =========================================================
        {
            std::unique_lock<std::mutex> lockImg(bufImgMutex_);

            if (imgBuf_.empty()) {
                lockImg.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            img_msg = imgBuf_.front();
            tImg_cam = Utility::StampToSec(img_msg->header.stamp);
            tImg = tImg_cam + kalibr_timeshift;
        }

        RCLCPP_INFO(this->get_logger(),
            "\n================ FRAME ================\n"
            "t_cam=%.6f t_aligned=%.6f",
            tImg_cam, tImg);

        // =========================================================
        // 2. IMAGE VALIDATION (DO NOT BLOCK ON IMU)
        // =========================================================
        {
            std::unique_lock<std::mutex> lockImg(bufImgMutex_);
            if (!imgBuf_.empty() && imgBuf_.front() == img_msg) {
                Img = GetImage(imgBuf_.front());
                imgBuf_.pop();
            }
        }

        if (Img.empty()) {
            RCLCPP_WARN(this->get_logger(), "Imagem vazia, pulando frame");
            continue;
        }

        // =========================================================
        // 3. BUILD IMU PACKET (SAFE CONSUMPTION)
        // =========================================================
        std::vector<ORB_SLAM3::IMU::Point> vImuMeas;

        {
            std::unique_lock<std::mutex> lockImu(bufImuMutex_);

            if (imuBuf_.empty()) {
                RCLCPP_WARN(this->get_logger(), "IMU buffer vazio");
            }

            double tPrev = -1.0;
            size_t idx = 0;

            while (!imuBuf_.empty())
            {
                auto imu = imuBuf_.front();
                double t = Utility::StampToSec(imu->header.stamp);

                // não avançar além da imagem
                if (t > tImg)
                    break;

                if (std::isnan(t)) {
                    RCLCPP_ERROR(this->get_logger(), "IMU timestamp NaN -> drop");
                    imuBuf_.pop();
                    continue;
                }

                if (tPrev > 0.0) {
                    double dt = t - tPrev;

                    RCLCPP_INFO(this->get_logger(),
                        "[IMU %zu] dt=%.9f t=%.6f",
                        idx, dt, t);

                    if (!std::isfinite(dt) || dt <= 0.0) {
                        RCLCPP_ERROR(this->get_logger(),
                            "DT inválido -> reset IMU stream");
                        imuBuf_.pop();
                        continue;
                    }

                    if (dt > 0.02) {
                        RCLCPP_WARN(this->get_logger(),
                            "DT grande: %.6f", dt);
                    }
                }

                // =========================
                // IMU conversion
                // =========================
                float ax = imu->linear_acceleration.x;
                float ay = imu->linear_acceleration.y;
                float az = imu->linear_acceleration.z;

                float gx = imu->angular_velocity.x;
                float gy = imu->angular_velocity.y;
                float gz = imu->angular_velocity.z;

                float scale = 9.81f / 9.8425f;

                cv::Point3f acc(ax * scale, ay * scale, az * scale);
                cv::Point3f gyr(gx, gy, gz);

                vImuMeas.emplace_back(acc, gyr, t);

                tPrev = t;
                imuBuf_.pop();
                idx++;
            }

            if (!vImuMeas.empty()) {
                RCLCPP_INFO(this->get_logger(),
                    "IMU PACKET FINAL: %zu samples | %.6f → %.6f",
                    vImuMeas.size(),
                    vImuMeas.front().t,
                    vImuMeas.back().t);
            }
        }

        // =========================================================
        // 4. BASIC VALIDATION BEFORE SLAM
        // =========================================================
        if (vImuMeas.size() < 2) {
            RCLCPP_WARN(this->get_logger(), "IMU insuficiente para frame");
            continue;
        }

        double dt_img = (tLastImg > 0.0) ? (tImg - tLastImg) : 0.0;

        RCLCPP_INFO(this->get_logger(),
            "TRACK INPUT: dt_img=%.6f imu=%zu",
            dt_img, vImuMeas.size());

        // =========================================================
        // 5. CALL SLAM
        // =========================================================
        try {
            m_SLAM->TrackMonocular(Img, tImg, vImuMeas);
            tLastImg = tImg;

            if (m_SLAM->GetTrackingState() == 2) {
                RCLCPP_INFO(this->get_logger(), "TRACKING OK");
            }
        }
        catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "SLAM exception: %s", e.what());
        }
        catch (...) {
            RCLCPP_ERROR(this->get_logger(), "unknown SLAM exception");
        }
    }
}

/*
//Solução do Gemini:

void MonoInertialNode::SyncWithImu_Track()
{   
    while(rclcpp::ok())
    {
        RCLCPP_INFO(this->get_logger(), "Iteração de SyncWithImu_Track chamada");
        if (!imgBuf_.empty() && !imuBuf_.empty()) {
                RCLCPP_INFO(this->get_logger(), "1) Buffers não estão vazios: Tempo Imagem: %f | Última IMU: %f", Utility::StampToSec(imgBuf_.front()->header.stamp), 
                Utility::StampToSec(imuBuf_.back()->header.stamp));
        }
        cv::Mat Img;
        double tImg = 0.0;
        ImageMsg::SharedPtr img_msg_ponteiro = nullptr;
        
        // 1. ESCOPO ISOLADO: Apenas verifica se há imagens e espia o timestamp
        {
            std::unique_lock<std::mutex> lockImg(bufImgMutex_);
            if (imgBuf_.empty()){
                lockImg.unlock();
                RCLCPP_INFO(this->get_logger(), "2) Buffer de imagens vazio, dando continue...");
                //std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            // Apenas pegamos o ponteiro e o tempo, NÃO damos pop() ainda!
            img_msg_ponteiro = imgBuf_.front();
            tImg = Utility::StampToSec(img_msg_ponteiro->header.stamp);
        } // O lockImg é destruído e liberado aqui automaticamente

        
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        
        // 2. ESCOPO ISOLADO: Sincronização temporal da IMU
        {
            std::unique_lock<std::mutex> lockImu(bufImuMutex_);

            // Se a IMU estiver vazia OU o último dado da IMU ainda for mais antigo que a imagem,
            if (imuBuf_.empty() || Utility::StampToSec(imuBuf_.back()->header.stamp) <= tImg)
            {
                lockImu.unlock();
                RCLCPP_INFO(this->get_logger(), "3) Buff imu vazio ou medida IMU mais nova ainda é mais antiga que a imagem, dando continue...");
                //std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue; // Volta ao topo e tenta ler A MESMA imagem de novo
            }

            // Se passou do IF acima, temos dados suficientes de IMU para cobrir a imagem.
            // Vamos descarregar o buffer da IMU até o tempo da imagem.
            vImuMeas.clear();
            while(!imuBuf_.empty() && Utility::StampToSec(imuBuf_.front()->header.stamp) <= tImg)
            {
                double t = Utility::StampToSec(imuBuf_.front()->header.stamp);
                cv::Point3f acc(imuBuf_.front()->linear_acceleration.x, imuBuf_.front()->linear_acceleration.y, imuBuf_.front()->linear_acceleration.z);
                cv::Point3f gyr(imuBuf_.front()->angular_velocity.x, imuBuf_.front()->angular_velocity.y, imuBuf_.front()->angular_velocity.z);
                
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                
                // Print de debug para validar unidades e sinais no terminal
                RCLCPP_INFO(this->get_logger(), "4) Mensagens imu parecem ser suficientes, preenchendo vetor vImuMeas: Acc: [%f, %f, %f] | Gyr: [%f, %f, %f]", 
                            acc.x, acc.y, acc.z, gyr.x, gyr.y, gyr.z);
                            
                imuBuf_.pop();
            }

            // Adiciona a primeira mensagem do "futuro" sem dar pop (exigência do ORB-SLAM3 para envelopar a imagem)
            if(!imuBuf_.empty())
            {
                double t = Utility::StampToSec(imuBuf_.front()->header.stamp);
                cv::Point3f acc(imuBuf_.front()->linear_acceleration.x, imuBuf_.front()->linear_acceleration.y, imuBuf_.front()->linear_acceleration.z);
                cv::Point3f gyr(imuBuf_.front()->angular_velocity.x, imuBuf_.front()->angular_velocity.y, imuBuf_.front()->angular_velocity.z);
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                RCLCPP_INFO(this->get_logger(), "5) Medida IMU mais nova que a imagem adicionada");
            }
        } // O lockImu é liberado aqui automaticamente

        // 3. Agora que a IMU está garantida e salva no vetor local,
        // retiramos (pop) a imagem do buffer com segurança fora de qualquer outra trava perigosa
        {
            std::unique_lock<std::mutex> lockImg(bufImgMutex_);
            if(!imgBuf_.empty() && imgBuf_.front() == img_msg_ponteiro) {
                Img = GetImage(imgBuf_.front());
                imgBuf_.pop();
            }
        }

        if(!vImuMeas.empty() && !Img.empty()) {
            
            // PROTEÇÃO: O ORB-SLAM3 precisa de pelo menos 2 pontos para calcular o delta de tempo (t_atual - t_anterior)
            // Para o primeiro frame, idealmente queremos uma janela robusta.
            if(vImuMeas.size() < 5) {
                RCLCPP_WARN(this->get_logger(), "6) Vetor de IMU muito pequeno (%lu pontos). Aguardando mais dados...", vImuMeas.size());
                continue;
            }

            double tUltimaImu = vImuMeas.back().t;
            double tPrimeiraImu = vImuMeas.front().t;
            double difTempos = tUltimaImu-tImg;
            RCLCPP_INFO(this->get_logger(), "7) Passamos pela sincronização com IMU. Enviando %lu pontos.", vImuMeas.size());
            RCLCPP_INFO(this->get_logger(), "7.5) Tempos dos dados: tImagem: %f, tPrimeiraImu: %f, tÚltimaImu: %f, diferença de tempos: %f", tImg, tPrimeiraImu, tUltimaImu, difTempos);
            try{
            Sophus::SE3f Tcm = m_SLAM->TrackMonocular(Img, tImg, vImuMeas);
            RCLCPP_INFO(this->get_logger(), "8) TrackMonocular chamado com sucesso. Estado do tracking: %d", m_SLAM->GetTrackingState());

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

/*void MonoInertialNode::SyncWithImu_Track()
{   
    
    while(rclcpp::ok()) //Sempre rodando, enquanto o nó estiver rodando.
    {
        RCLCPP_INFO(this->get_logger(), "Iteração de SyncWithImu_Track chamada");
        cv::Mat Img;
        
        bufImgMutex_.lock();
        if (imgBuf_.empty()){
            bufImgMutex_.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        double tImg = Utility::StampToSec(imgBuf_.front()->header.stamp);
        Img = GetImage(imgBuf_.front());
        imgBuf_.pop();
        bufImgMutex_.unlock();

        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        bufImuMutex_.lock();

        if (!imuBuf_.empty())
        {
            // Load imu measurements from buffer
            vImuMeas.clear();
            while (!imuBuf_.empty() && Utility::StampToSec(imuBuf_.front()->header.stamp) <= tImg)
            {
                double t = Utility::StampToSec(imuBuf_.front()->header.stamp);
                cv::Point3f acc(imuBuf_.front()->linear_acceleration.x, imuBuf_.front()->linear_acceleration.y, imuBuf_.front()->linear_acceleration.z);
                cv::Point3f gyr(imuBuf_.front()->angular_velocity.x, imuBuf_.front()->angular_velocity.y, imuBuf_.front()->angular_velocity.z);
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                imuBuf_.pop();
            }
        }

        bufImuMutex_.unlock();

        if(!vImuMeas.empty()) {
            RCLCPP_INFO(this->get_logger(), "Passamos pela sincronização com IMU");
            Sophus::SE3f Tcm = m_SLAM->TrackMonocular(Img, tImg, vImuMeas); //Tracking do orbslam3
            RCLCPP_INFO(this->get_logger(), "TrackMonocular chamado");
        }

        /*Sophus::SE3f Tmc = Tcm.inverse(); //Transformação mapa => camera (está em base_link pela calibração do slam)
        
        TfMsg transf_msg;
        try {
            TfMsg odom_to_base_msg = tf_buffer_->lookupTransform("odom", "base_link", tf2::TimePointZero);
            auto translation = odom_to_base_msg.transform.translation;
            auto rotation = odom_to_base_msg.transform.rotation;
            Eigen::Vector3f trans(translation.x, translation.y, translation.z);
            Eigen::Quaternionf rot(rotation.w, rotation.x, rotation.y, rotation.z);
            Sophus::SE3f Tob(rot, trans);
            Sophus::SE3f Tbo = Tob.inverse();
            Sophus::SE3f Tmo = Tmc * Tbo; //Transformação mapa => odometria, que o nav2 requisita

            Eigen::Quaternionf Tmo_q(Tmo.rotationMatrix());
            transf_msg.transform.translation.x = Tmo.translation().x();
            transf_msg.transform.translation.x = Tmo.translation().y();
            transf_msg.transform.translation.x = Tmo.translation().z();
            transf_msg.transform.rotation.w = Tmo_q.w();
            transf_msg.transform.rotation.x = Tmo_q.x();
            transf_msg.transform.rotation.y = Tmo_q.y();
            transf_msg.transform.rotation.z = Tmo_q.z();

            transf_msg.header.stamp = this->get_clock()->now();
            transf_msg.header.frame_id = "map";
            transf_msg.child_frame_id = "odom";
            tf_broadcaster_->sendTransform(transf_msg); //Publica transformação pelo tf2
        } catch (const tf2::TransformException & ex) {
            RCLCPP_INFO( this->get_logger(), "Could not find odom to base_link transform");
            return;
        }
        //TODO: Talvez precise colocar um sleep igual o q tem em stereo-inertial.
        /
    }
}*/