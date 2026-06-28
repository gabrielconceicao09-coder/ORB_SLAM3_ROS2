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
//Solução 2 do Gemini:

void MonoInertialNode::SyncWithImu_Track()
{   
    // Guarda o timestamp da última imagem processada com sucesso para garantir continuidade perfeita
    double tLastImg = -1.0;

    while(rclcpp::ok())
    {
        RCLCPP_INFO(this->get_logger(), "Iteração de SyncWithImu_Track chamada");
        
        if (!imgBuf_.empty() && !imuBuf_.empty()) {
            RCLCPP_INFO(this->get_logger(), "1) Buffers não estão vazios: Tempo Imagem: %f | Última IMU: %f", 
                        Utility::StampToSec(imgBuf_.front()->header.stamp), 
                        Utility::StampToSec(imuBuf_.back()->header.stamp));
        }

        cv::Mat Img;
        double tImg = 0.0;
        ImageMsg::SharedPtr img_msg_ponteiro = nullptr;
        
        // 1. ESCOPO ISOLADO: Verifica se há imagens e recupera o timestamp (apenas espia)
        {
            std::unique_lock<std::mutex> lockImg(bufImgMutex_);
            if (imgBuf_.empty()){
                lockImg.unlock();
                RCLCPP_INFO(this->get_logger(), "2) Buffer de imagens vazio, dando continue...");
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            img_msg_ponteiro = imgBuf_.front();
            tImg = Utility::StampToSec(img_msg_ponteiro->header.stamp);
        }

        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        
        // 2. ESCOPO ISOLADO: Sincronização temporal da IMU com proteção contra Delta T zero/negativo
        {
            std::unique_lock<std::mutex> lockImu(bufImuMutex_);

            // Condição essencial: Precisamos ter dados de IMU que alcancem ou passem o tempo da imagem atual
            if (imuBuf_.empty() || Utility::StampToSec(imuBuf_.back()->header.stamp) < tImg)
            {
                lockImu.unlock();
                RCLCPP_INFO(this->get_logger(), "3) Buff imu vazio ou medida IMU mais nova ainda é mais antiga que a imagem, dando continue...");
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue; 
            }

            // Se for o primeiro frame absoluto, precisamos garantir que temos dados de IMU ANTES da imagem (histórico)
            if (tLastImg < 0.0) {
                if (Utility::StampToSec(imuBuf_.front()->header.stamp) >= tImg) {
                    lockImu.unlock();
                    std::unique_lock<std::mutex> lockImg(bufImgMutex_);
                    if(!imgBuf_.empty()) imgBuf_.pop();
                    RCLCPP_WARN(this->get_logger(), "Descartando frame de imagem inicial sem histórico de IMU anterior.");
                    continue;
                }
            }

            vImuMeas.clear();

            // Fator de correção puro do seu hardware para a escala terrestre (9.81 / 10.9391)
            double fator_escala_acc = 9.81 / 10.9391;
            double tLastImuInPacket = -1.0;

            // Descarrega os pontos da IMU até o tempo da imagem
            while(!imuBuf_.empty() && Utility::StampToSec(imuBuf_.front()->header.stamp) <= tImg)
            {
                double t = Utility::StampToSec(imuBuf_.front()->header.stamp);
                
                // PROTEÇÃO CRÍTICA CONTRA DELTA-T NULO OU INVERTIDO (Evita NaN no Sophus)
                if (tLastImuInPacket >= 0.0 && t <= tLastImuInPacket) {
                    // Se o carimbo de tempo for idêntico ou menor (jitter de rede), força um avanço artificial coerente (200Hz -> 5ms)
                    t = tLastImuInPacket + 0.005; 
                }

                // Mantemos a ordem original dos eixos conforme calibrado pelo Kalibr
                cv::Point3f acc(
                    imuBuf_.front()->linear_acceleration.x * fator_escala_acc, 
                    imuBuf_.front()->linear_acceleration.y * fator_escala_acc, 
                    imuBuf_.front()->linear_acceleration.z * fator_escala_acc 
                );
                cv::Point3f gyr(
                    imuBuf_.front()->angular_velocity.x, 
                    imuBuf_.front()->angular_velocity.y, 
                    imuBuf_.front()->angular_velocity.z
                );
                
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                tLastImuInPacket = t; // Atualiza a âncora de tempo do último elemento adicionado

                // Regra de Ouro: Se for o último elemento menor ou igual a tImg, NÃO damos pop.
                // Ele precisa ficar na fila para servir de início (ponto de ancoragem) para o próximo frame.
                if (imuBuf_.size() > 1) {
                    imuBuf_.pop();
                } else {
                    break; 
                }
            }

            // Adiciona um ponto do "futuro imediato" (o próximo após tImg) para fechar o envelope temporal
            if(!imuBuf_.empty())
            {
                double t = Utility::StampToSec(imuBuf_.front()->header.stamp);
                
                // Aplicamos a proteção de consistência temporal também para o ponto do futuro
                if (tLastImuInPacket >= 0.0 && t <= tLastImuInPacket) {
                    t = tLastImuInPacket + 0.005;
                }

                if (t > tImg) {
                    cv::Point3f acc(
                        0.0f,//imuBuf_.front()->linear_acceleration.x * fator_escala_acc, 
                        0.0f,//imuBuf_.front()->linear_acceleration.y * fator_escala_acc, 
                        -9.81f//imuBuf_.front()->linear_acceleration.z * fator_escala_acc
                    );
                    cv::Point3f gyr(
                        0.0f,//imuBuf_.front()->angular_velocity.x, 
                        0.0f,//imuBuf_.front()->angular_velocity.y, 
                        0.0f//imuBuf_.front()->angular_velocity.z
                    );
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                    RCLCPP_INFO(this->get_logger(), "5) Medida IMU mais nova que a imagem (futuro) adicionada.");
                }
            }
        } 

        // 3. Extrai e remove a imagem do buffer de forma segura
        {
            std::unique_lock<std::mutex> lockImg(bufImgMutex_);
            if(!imgBuf_.empty() && imgBuf_.front() == img_msg_ponteiro) {
                Img = GetImage(imgBuf_.front());
                imgBuf_.pop();
            }
        }

        // 4. Executa o Tracking se os pacotes estiverem íntegros e sincronizados
        if(!vImuMeas.empty() && !Img.empty()) {
            
            // O ORB-SLAM3 precisa de pelo menos 2 pontos para calcular o delta de tempo interno
            if(vImuMeas.size() < 2) {
                RCLCPP_WARN(this->get_logger(), "6) Vetor de IMU muito pequeno (%lu pontos). Aguardando mais dados...", vImuMeas.size());
                continue;
            }

            double tUltimaImu = vImuMeas.back().t;
            double tPrimeiraImu = vImuMeas.front().t;
            double difTempos = tUltimaImu - tImg;
            
            RCLCPP_INFO(this->get_logger(), "7) Passamos pela sincronização com IMU. Enviando %lu pontos.", vImuMeas.size());
            RCLCPP_INFO(this->get_logger(), "7.5) Tempos dos dados: tImagem: %f, tPrimeiraImu: %f, tÚltimaImu: %f, diferença de tempos: %f", 
                        tImg, tPrimeiraImu, tUltimaImu, difTempos);
            
            try {
                // Envia a imagem e o vetor contínuo blindado de IMU para o motor do SLAM
                Sophus::SE3f Tcm = m_SLAM->TrackMonocular(Img, tImg, vImuMeas);
                
                tLastImg = tImg; // Salva o tempo atual como histórico de sucesso
                
                RCLCPP_INFO(this->get_logger(), "8) TrackMonocular chamado com sucesso. Estado do tracking: %d", m_SLAM->GetTrackingState());
                
                int estado = m_SLAM->GetTrackingState();
                if(estado == 2) {
                    RCLCPP_INFO(this->get_logger(), "=== TRACKING OK (ESTADO 2) ===");
                }

            } catch (...) {
                RCLCPP_ERROR(this->get_logger(), "Algum problema ou crash interno ocorreu dentro do TrackMonocular!");
                continue;
            }
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