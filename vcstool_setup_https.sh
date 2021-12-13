mkdir -p src/dependency
vcs import src/dependency < src/Interactive-Scene-Reconstruction/mapping/orb_slam_2_ros/orb_slam2_ros_https.rosinstall
vcs import src/dependency < src/Interactive-Scene-Reconstruction/mapping/perception_ros/perception_ros_https.rosinstall
vcs import src/dependency < src/Interactive-Scene-Reconstruction/mapping/voxblox-plusplus/voxblox-plusplus_https.rosinstall
vcs import src/dependency < src/Interactive-Scene-Reconstruction/cad_replacement/map_proc/map_proc_https.rosinstall
vcs pull src/dependency
