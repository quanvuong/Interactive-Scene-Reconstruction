#!/usr/bin/env bash
# Usage: ./scene_recon_with_models.sh <path-to-rosbag-dir>

############################################################
# Section 0: Script-Specific Settings                      #
############################################################

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

check_roslaunch_status() {
  # Check if roslaunch starts successfully
  if ps -p "$ROSLAUNCH_PID" > /dev/null ; then
    echo -e "\nroslaunch process ${ROSLAUNCH_PID} started...\n"
  else
    echo -e "\nroslaunch process ${ROSLAUNCH_PID} is not running... Exiting...\n"
    exit 2
  fi
}

save_roslaunch_log() {
  if [[ ! -d "$ROSLAUNCH_LOG_DIR" ]] ; then
    echo -e "\nLog directory ${ROSLAUNCH_LOG_DIR} does not exist... Exiting...\n"
    exit 3
  fi
  SAVE_LOG_DIR="${SCRIPTPATH}/../output/${SAVE_DIR_NAME}/${EXP_NAME}/${1}"
  mkdir -p "$SAVE_LOG_DIR"
  test_retval "create log saving directory ${SAVE_LOG_DIR}"
  cp -r "$ROSLAUNCH_LOG_DIR"/* "$SAVE_LOG_DIR"
  test_retval "moving ROS log directory ${ROSLAUNCH_LOG_DIR}"
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
  echo -e "\nWrong number of arguments: ${#}. Expects only 1 directory... Exiting...\n"
  exit 4
fi
if [[ ! -d "$1" ]] ; then
  echo -e "\nDirectory ${1} does not exist... Exiting...\n"
  exit 5
fi

. "${SCRIPTPATH}/../../../devel/setup.bash"  # sourcing ROS project setup.bash

SAVE_DIR_NAME=$(basename "$1")

for file in "$1"/*.bag; do
  echo "Start mapping from ${file} ..."
  EXP_NAME=$(basename "$file" .bag)

  roslaunch panoptic_mapping_pipeline mobile_pnp_pano_mapping.launch \
            sequence:="${SAVE_DIR_NAME}/${EXP_NAME}" \
            visualize_fusion_seg:=false \
            output:=log &
  ROSLAUNCH_PID="$!"
  sleep 5  # sleep briefly for ROS
  check_roslaunch_status  # Check if roslaunch starts successfully
  ROSLAUNCH_LOG_DIR="$(roslaunch-logs)"

  rosbag play -r 0.02 "$file"  # ~2 fps
  test_retval "playback rosbag file ${file}"
  sleep 10  # sleep briefly for ROS

  rosservice call /gsm_node/generate_mesh
  test_retval "rosservice call /gsm_node/generate_mesh"
  rosservice call /gsm_node/extract_instances
  test_retval "rosservice call /gsm_node/extract_instances"
  sleep 5  # sleep briefly for ROS

  kill "$ROSLAUNCH_PID"
  sleep 5  # sleep briefly for ROS
  save_roslaunch_log "pano_mapping_log"

  roslaunch map_proc mobile_pnp_map_processing.launch \
            sequence:="${SAVE_DIR_NAME}/${EXP_NAME}" \
            visualize_scene_mesh:=false visualize_global_regulation:=false \
            visualize_final_scene:=false output:=log
  sleep 5  # sleep briefly for ROS
  ROSLAUNCH_LOG_DIR="$(roslaunch-logs)"/latest  # manually get log dir
  save_roslaunch_log "map_proc_log"
done

echo -e "\nScript ${0} finished successfully\n"
