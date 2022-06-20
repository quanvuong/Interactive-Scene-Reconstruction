#!/usr/bin/env bash
# Usage: ./view_rosbag.sh <path-to-rosbag-file>

############################################################
# Section 0: Script-Specific Settings                      #
############################################################
IMAGE_TOPIC="/camera/rgb/image_raw"
#IMAGE_TOPIC="/camera/depth/image_raw"

############################################################
# Section 1: Helper Function Definition                    #
############################################################
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
#cd "$SCRIPTPATH"

test_retval() {
  if [ $? -ne 0 ] ; then
    echo -e "\nFailed to ${*}... Exiting...\n"
    echo "Log is saved to ${ROSLAUNCH_LOG_DIR}"
    exit 1
  fi
}

trap_sigint() {
  kill "$ROSLAUNCH_PID"
  exit 0
}

############################################################
# Section 2: Invoke script                                 #
############################################################
### Verify command line arguments ###
if [ $# -ne 1 ] ; then
  echo -e "\nWrong number of arguments: ${#}. Expects only 1 rosbag... Exiting...\n"
  exit 1
fi
if [[ ! -f "$1" ]] ; then
  echo -e "\nrosbag file ${1} does not exist... Exiting...\n"
  exit 1
fi

roslaunch "$SCRIPTPATH"/image_view.launch image_topic:="$IMAGE_TOPIC" &
ROSLAUNCH_PID="$!"
sleep 1  # sleep briefly for ROS
### Check if roslaunch starts successfully ###
if ps -p "$ROSLAUNCH_PID" > /dev/null ; then
  echo -e "\nroslaunch process ${ROSLAUNCH_PID} started...\n"
else
  echo -e "\nroslaunch process ${ROSLAUNCH_PID} is not running... Exiting...\n"
  exit 1
fi
ROSLAUNCH_LOG_DIR="$(roslaunch-logs)"

rosbag play -r 0.02 "$1"  # ~2 fps
#rosbag play "$1"
test_retval "playback rosbag file ${1}"

trap 'trap_sigint' SIGINT
sleep infinity

#kill "$ROSLAUNCH_PID"
