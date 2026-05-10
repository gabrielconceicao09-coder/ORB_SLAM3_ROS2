
#include "mono-inertial-node.hpp"

#include<opencv2/core/core.hpp>

using std::placeholders::_1;

MonoInertialNode::MonoInertialNode(ORB_SLAM3::System* pSLAM)
:   Node("ORB_SLAM3_ROS2")
{
    m_SLAM = pSLAM;
    // std::cout << "slam changed" << std::endl;
    m_image_subscriber = this->create_subscription<ImageMsg>(
        "camera",
        10,
        std::bind(&MonoInertialNode::GrabImage, this, std::placeholders::_1));
    
    imu_subscriber = this->create_subscription<ImuMsg>(
        "imu",
        1000,
        std::bind(&MonoInertialNode::GrabImu, this, std::placeholders::_1));
    );

    pointcloud_publisher = this->create_publisher<PclMsg>(
        "PCLTOPIC"
        10
    );

    /*timer_ = this->create_wall_timer(
        20ms,
        std::bind(&MonoInertialNode::timer_callback, this) // o timer_ é chamado pelo node::spin() e ta bindado com o método timer_callback
    )*/

    syncThread_ = new std::thread(&MonoInertialNode::SyncWithImu_Track(), this);

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
    //Não entendi muito bem, mas de alguma forma esse Mutex impede que outras threads alterem imuBuff_ ao mesmo tempo
    bufImuMutex_.lock();
    imuBuf_.push(msg);
    bufImuMutex_.unlock();
}

void MonoInertialNode::GrabImage(const ImageMsg::SharedPtr msg)
{
    bufImgMutex_.lock();
    imgBuf_.push(msg);
    bufImgMutex_.lock();
}

cv::Mat MonoInertialNode::GetImage(const ImageMsg::SharedPtr msg)
{

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
        std::cerr << "Error image type" << std::endl;
        return m_cvImPtr->image.clone();
    }
    /* Aqui não é o lugar pra chamar. Tem que sincronizar o Imu com as Imagens antes.
    std::cout<<"one frame has been sent"<<std::endl;
    m_SLAM->TrackMonocular(m_cvImPtr->image, Utility::StampToSec(msg->header.stamp));
    */
}

void MonoInertialNode::SyncWithImu_Track()
{   
    
    while(1) //Sempre rodando, i guess
    {
        cv::Mat Img;

        double tImg = Utility::StampToSec(imgBuf_.front()->header.stamp);

        bufImgMutex_.lock();
        Img = GetImage(imgBuf_.front());
        imgBuf_.pop();
        bufImgMutex_.unlock();

        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        bufImuMutex_.lock();
        if (!imuBuf_.empty())
        {
            //Load imu measurements from buffer
            vImuMeas.clear();
            while(!imuBuf_.empty() && Utility::StampToSec(imuBuf_.front()->header.stamp) <= tImg)
            {
                double t = Utility::StampToSec(imuBuf_.front()->header.stamp);
                cv::Point3f acc(imuBuf_.front()->linear_acceleration.x, imuBuf_.front()->linear_acceleration.y, imuBuf_.front()->linear_acceleration.z);
                cv::Point3f gyr(imuBuf_.front()->angular_velocity.x, imuBuf_.front()->angular_velocity.y, imuBuf_.front()->angular_velocity.z);
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                imuBuf_.pop();
            }
        }
        bufImuMutex_.unlock();
        
        Sophus::SE3f Tcm = m_SLAM->TrackMonocular(Img, tImg, vImuMeas); //Tracking do orbslam3
        
        Sophus::SE3f Tmc = Tcm.inverse();
        tf2::Transform Tmc_tf //Montando transformada tf2 a partir do SE3f Tmc
        Tmc_tf.rotation.x = Tmc.translation().x();
        Tmc_tf.rotation.y = Tmc.translation().y();
        Tmc_tf.rotation.z = Tmc.translation().z();
        Tmc_tf.orientation.w = Tmc.unit_quaternion().coeffs().w();
        Tmc_tf.orientation.x = Tmc.unit_quaternion().coeffs().x();
        Tmc_tf.orientation.y = Tmc.unit_quaternion().coeffs().y();
        Tmc_tf.orientation.z = Tmc.unit_quaternion().coeffs().z();
        
        TfMsg transf_msg;
        try {
            TfMsg odom_to_base_msg = tf_buffer_->lookupTransform("odom", "base_link", tf2::TimePointZero);
            tf2::Transform Tmap_odom = Tmc_tf * odom_to_base_msg.transform.inverse(); //Calculando transformação map => odom pedida pelo nav2
            transf_msg.transform = Tmap_odom;
            transf_msg.header.stamp = this->get_clock()->now();
            transf_msg.header.frame_id = "map";
            transf_msg.child_frame_id = "odom";
            tf_broadcaster_->sendTransform(transf_msg); //Publica transformação pelo tf2
        } catch (const tf2::TransformException & ex) {
            RCLCPP_INFO( this->get_logger(), "Could not find odom to base_link transform");
            return;
        }
        bufTfMutex_.lock(); //Coloca uma tf coletada no buffer
        tfBuf_.push(curr_Tf);
        bufTfMutex_.unlock();
        //TODO: Talvez precise colocar um sleep igual o q tem em stereo-inertial.

    }
}

/*TfMsg MonoInertialNode::MakeTfmsg(const Sophus::SE3f::SharedPtr Tcm)
{
    Sophus::SE3f Tmc = Tcm.inverse();
    tf2::Transform Tmc_tf //Montando transformada tf2 a partir do SE3f Tmc
    Tmc_tf.rotation.x = Tmc.translation().x();
    Tmc_tf.rotation.y = Tmc.translation().y();
    Tmc_tf.rotation.z = Tmc.translation().z();
    Tmc_tf.orientation.w = Tmc.unit_quaternion().coeffs().w();
    Tmc_tf.orientation.x = Tmc.unit_quaternion().coeffs().x();
    Tmc_tf.orientation.y = Tmc.unit_quaternion().coeffs().y();
    Tmc_tf.orientation.z = Tmc.unit_quaternion().coeffs().z();
    
    TfMsg transf_msg;
    try {
        odom_to_base_msg = tf_buffer_->lookupTransform("odom", "base_link", tf2::TimePointZero);
        tf2::Transform Tmap_odom = Tmc_f * odom_to_base_msg.Transform.inverse(); //Calculando transformada map => odom pedida pelo nav2
        transf_msg.Transform = Tmap_odom;
        transf_msg.frame_id = "map";
        transf_msg.child_frame_id = "odom";
        return transf_msg;
    } catch (const tf2::TransformException & ex) {
        RCLCPP_INFO( this->get_logger(), "Could not find odom to base_link transform");
        return;
    }
    
}*/

void MonoInertialNode::timer_callback()
{
    Sophus::SE3f::SharedPtr Tf_;
    TfMsg msg;

    bufTfMutex_.lock(); //copiei a estratégia buffer e lock com mutex
    Tf_ = tfBuf_.front();
    //msg = MakeTfMsg(Tf_);
    tfBuf_.pop();
    bufTfMutex_.unlock();

    tf_broadcaster->sendTransform(msg);
}
