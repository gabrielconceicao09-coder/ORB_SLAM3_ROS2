
#include "monocular-inertial-slam-node.hpp"

#include<opencv2/core/core.hpp>

using std::placeholders::_1;

MonocularInertialSlamNode::MonocularInertialSlamNode(ORB_SLAM3::System* pSLAM)
:   Node("ORB_SLAM3_ROS2")
{
    m_SLAM = pSLAM;
    // std::cout << "slam changed" << std::endl;
    m_image_subscriber = this->create_subscription<ImageMsg>(
        "camera",
        10,
        std::bind(&MonocularInertialSlamNode::GrabImage, this, std::placeholders::_1));
    
    imu_subscriber = this->create_subscription<ImuMsg>(
        "imu",
        1000,
        std::bind(&MonocularInertialSlamNode::GrabImu, this, std::placeholders::_1));
    );

    twc_publisher = this->create_publisher<TfMsg>(
        "tf",
        10
    );

    pcl_publisher = this->create_publisher<PclMsg>(
        "PCLTOPIC"
        10
    );

    timer_ = this->create_wall_timer(
        20ms,
        std::bind(&MonocularInertialSlamNode::timer_callback, this) // o timer_ é chamado pelo node::spin() e ta bindado com o método timer_callback
    )

    syncThread_ = new std::thread(&StereoInertialNode::SyncWithImu, this);

    std::cout << "slam changed" << std::endl;
}

MonocularInertialSlamNode::~MonocularInertialSlamNode()
{
    //Delete sync thread
    syncThread_->join();
    delete syncThread_;

    // Stop all threads
    m_SLAM->Shutdown();

    // Save camera trajectory
    m_SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}

void MonocularInertialSlamNode::GrabImu(const ImuMsg::SharedPtr msg)
{
    //Não entendi muito bem, mas de alguma forma esse Mutex impede que outras threads alterem imuBuff_ ao mesmo tempo
    buffImuMutex_.lock();
    imuBuf_.push(msg);
    bufImuMutex.unlock();
}

void MonocularInertialSlamNode::GrabImage(const ImageMsg::SharedPtr msg)
{
    bufImgMutex_.lock();
    imgBuf_.push(msg);
    bufImgMutex_.lock();
}

cv::Mat MonocularInertialSlamNode::GetImage(const ImageMsg::SharedPtr msg)
{

    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr m_cvImPtr;

    try
    {
        m_cvImPtr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
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

Sophus::SE3f MonocularInertialSlamNode::SyncWithImu_Track()
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
        bufMutex_.lock();
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
        bufMutex_.unlock();
        
        Sophus::SE3f curr_Tf;
        curr_Tf = m_SLAM->TrackMonocular(Img, tImg, vImuMeas);

        bufTfMutex_.lock(); //Coloca uma tf coletada no buffer
        tfBuf_.push(curr_Tf);
        bufTfMutex_.unlock();
        //TODO: Talvez precise colocar um sleep igual o q tem em stereo-inertial.

    }
}

TfMsg MonocularInertialSlamNode::MakeTfMsg(const Sophus::SE3f::SharedPtr Tcm)
{
    Sophus::SE3f Tmc = Tcm.inverse();
    TfMsg transf_msg;

    transf_msg.transform.rotation.x = Tmc.translation().x();
    transf_msg.transform.rotation.y = Tmc.translation().y();
    transf_msg.transform.rotation.z = Tmc.translation().z();

    transf_msg.transform.orientation.w = Tmc.unit_quaternion().coeffs().w();
    transf_msg.transform.orientation.x = Tmc.unit_quaternion().coeffs().x();
    transf_msg.transform.orientation.y = Tmc.unit_quaternion().coeffs().y();
    transf_msg.transform.orientation.z = Tmc.unit_quaternion().coeffs().z();

    transf_msg.frame_id = "map";
    transf_msg.child_frame_id = "camera";
    return transf_msg;
}

void MonocularInertialSlamNode::timer_callback()
{
    Sophus::SE3f::SharedPtr Tf_;
    TfMsg msg;

    bufTfMutex.lock(); //copiei a estratégia buffer e lock com mutex
    Tf_ = tfBuf_.front();
    msg = MakeTfMsg(Tf_);
    tfBuf_.pop();
    bufTfMutex_.unlock();

    tf_publisher->publish(msg);
}
