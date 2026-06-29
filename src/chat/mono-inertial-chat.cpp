#include "mono-inertial-chat.hpp"

#include <memory>

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<MonoInertialNode>();

    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(),
        2);   // callbacks + thread do ORB-SLAM3

    executor.add_node(node);

    executor.spin();

    executor.remove_node(node);

    rclcpp::shutdown();

    return 0;
}