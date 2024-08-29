source /opt/intel/openvino_2024/setupvars.sh 
source ./scripts/setup_env.sh
#source /opt/intel/dlstreamer/setupvars.sh 
cd ~/src/dlstreamer-wqiiqw/dlstreamer/samples/gstreamer/python/draw_face_attributes/
./draw_face_attributes.sh ./head-pose-face-detection-female-and-male.mp4 
