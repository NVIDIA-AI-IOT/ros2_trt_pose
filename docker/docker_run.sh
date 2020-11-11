sudo xhost +si:localuser:root
sudo docker run -it --rm --runtime nvidia --device="/dev/video0:/dev/video0" --network host -v /tmp/.X11-unix/:/tmp/.X11-unix/ -v /home/ak-nv/ros_docker:/workdir ros2_trt_pose_base:jp44
