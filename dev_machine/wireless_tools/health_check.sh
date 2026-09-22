#!/bin/bash
# READ-ONLY health check of the running wireless navigation stack. Prints OK / WARN / FAIL per item and a final verdict.
# usage: health_check.sh          (starts nothing, stops nothing)
TOOLS=$(cd "$(dirname "$0")" && pwd); W=/tmp/go2_stack; mkdir -p $W
J="ssh -o BatchMode=yes -o ConnectTimeout=6 unitree@192.168.3.18"
STK="source /opt/ros/jazzy/setup.bash; source /home/tommy/GO2_STACK_dev_ws/install/setup.bash; source /home/tommy/Taowen/autonomy_stack_go2/ros_env_wifi.sh >/dev/null 2>&1"
NF=0; NW=0
ok()   { printf "  OK    %s\n" "$1"; }
warn() { NW=$((NW+1)); printf "  WARN  %s\n" "$1"; }
fail() { NF=$((NF+1)); printf "  FAIL  %s\n" "$1"; }
num()  { echo "$1" | grep -oE '[0-9]+\.[0-9]+' | head -1; }
ge()   { python3 -c "import sys; sys.exit(0 if float('${1:-0}')>=float('$2') else 1)" 2>/dev/null; }

echo "-- processes (dev machine) --"
for pat in "[l]aser_mapping_node:SLAM" "[s]ensor_scan_generation:scan generation" "[t]errainAnalysis :terrain analysis" "[l]ocalPlanner:localPlanner" "[u]nitree_webrtc_ros/lib/unitree_webrtc_ros/unitree_control:unitree_control (WebRTC)"; do
  p=${pat%%:*}; n=${pat#*:}; [ "$(pgrep -f "$p" | wc -l)" -ge 1 ] && ok "$n running" || fail "$n NOT running"; done
[ "$(pgrep -x pathFollower | wc -l)" = 0 ] && ok "pathFollower not running (robot cannot be commanded to walk)" || warn "pathFollower IS running ($(pgrep -x pathFollower | wc -l))"
[ "$(ss -ltn | grep -cE ':1218[1-5] ')" -ge 5 ] && ok "VLM servers 5/5" || warn "VLM servers $(ss -ltn | grep -cE ':1218[1-5] ')/5 (main script needs 12182 + 12185)"
grep -q "Subscribed to" $W/webrtc.log 2>/dev/null && ok "WebRTC: unitree_control connected to the Go2" || warn "WebRTC connect line not found in $W/webrtc.log (stack started by hand?)"

echo "-- data flow --"
OD=$(bash -c "unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH; $STK; /usr/bin/python3 $TOOLS/odom_monitor.py 10" 2>&1 | grep -E "^/laser_odometry:|^/livox/imu:")
R1=$(num "$(echo "$OD" | grep laser_odometry | grep -oE '= [0-9.]+ Hz')"); R2=$(num "$(echo "$OD" | grep livox/imu | grep -oE '= [0-9.]+ Hz')")
G1=$(echo "$OD" | grep laser_odometry | grep -oE 'gaps>0.5s: [0-9]+' | grep -oE '[0-9]+$')
if [ -z "$R1" ]; then fail "no /laser_odometry data (Livox/SLAM down, or wrong ROS domain)"; else ge "$R1" 8.5 && ok "/laser_odometry ${R1} Hz (expect 9-10)" || warn "/laser_odometry ${R1} Hz (< 8.5)"; [ "${G1:-0}" = 0 ] && ok "no odometry stall > 0.5 s" || warn "odometry stalls > 0.5 s: $G1"; fi
[ -z "$R2" ] && fail "no /livox/imu data" || { ge "$R2" 180 && ok "/livox/imu ${R2} Hz (expect ~190-200)" || warn "/livox/imu ${R2} Hz (< 180)"; }
TM=$(num "$(bash -c "$STK; timeout 8 ros2 topic hz /terrain_map --window 20 2>&1 | grep 'average rate' | tail -1")")
[ -z "$TM" ] && fail "no /terrain_map data" || { ge "$TM" 2.4 && ok "/terrain_map ${TM} Hz (2.4-3.5 is normal)" || warn "/terrain_map ${TM} Hz"; }
timeout 3 nc -z 192.168.3.18 6000 && ok "camera stream port 6000 open" || fail "camera stream port 6000 CLOSED (RealSense server down)"
CI=$(bash -c "$STK; timeout 15 ros2 topic info /cmd_vel --no-daemon 2>/dev/null | grep -E 'Publisher count|Subscription count'" | tr '\n' ' ')
echo "$CI" | grep -q "Publisher count: 0" && ok "/cmd_vel gate: $CI" || warn "/cmd_vel: ${CI:-topic not visible}"

echo "-- robot link / Jetson --"
GP=$(ping -c3 -W2 192.168.123.161 2>&1 | tail -1 | awk -F/ '{print $5}'); [ -n "$GP" ] && ok "Go2 reachable via Jetson relay, avg ${GP} ms" || fail "Go2 (192.168.123.161) not reachable"
JS=$($J 'echo "$(iw dev wlan0 link | awk "/signal:/{print \$2}") $(sudo dmesg | grep -c "URB .* submitted while active") $(cat /sys/devices/virtual/thermal/thermal_zone*/temp | sort -n | tail -1 | awk "{printf \"%.0f\", \$1/1000}") $(pgrep -f "[l]ivox_ros_driver2_node" | wc -l) $(pgrep -f "[r]ealsense_stream_server" | wc -l)"' 2>/dev/null)
set -- $JS
if [ -z "$1" ]; then fail "cannot read the Jetson over ssh"; else
  [ "$1" -gt -65 ] && ok "Jetson Wi-Fi signal $1 dBm" || warn "Jetson Wi-Fi signal weak: $1 dBm"
  [ "$2" = 0 ] && ok "Wi-Fi driver: 0 'URB submitted while active' warnings" || fail "Wi-Fi driver warnings: $2 (driver may be wedged -> see manual 'WiFi 驱动卡死的恢复')"
  [ "$3" -lt 80 ] && ok "Jetson temperature ${3} C" || warn "Jetson temperature ${3} C"
  [ "$4" -ge 1 ] && ok "Livox driver running on the Jetson" || fail "Livox driver NOT running on the Jetson"
  [ "$5" -ge 1 ] && ok "RealSense server running on the Jetson" || fail "RealSense server NOT running on the Jetson"
fi
echo "-- errors in this stack's logs (only if started by bringup_nav_stack.sh) --"
for f in slam scangen terrain terrainx localpl; do [ -f $W/$f.log ] && { c=$(grep -ciE '\[ERROR\]|process has died|Traceback|Segmentation' $W/$f.log); [ "$c" = 0 ] && ok "$f.log: 0 errors" || warn "$f.log: $c error lines"; }; done
echo; [ $NF = 0 ] && echo "VERDICT: READY  ($NW warning(s))" || echo "VERDICT: NOT READY  ($NF failure(s), $NW warning(s))"
