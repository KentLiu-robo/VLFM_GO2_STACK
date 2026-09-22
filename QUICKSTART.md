# QUICKSTART

完整拉起流程，对应 2026-09-19~22 当时实际跑通的裸机（不进 docker）部署方式。默认你已经有一台 Unitree GO2、一台机载 Jetson、一台通过网线连机器人的开发机。

## 0. 先把工作区组装起来

这个仓库**不是**一个可以直接 `colcon build` 的完整工作区——只包含本项目实际编写/修改的包。还需要额外 clone 这些第三方依赖：

```bash
mkdir -p ~/GO2_STACK_dev_ws/src && cd ~/GO2_STACK_dev_ws/src

# 1) 把本仓库 dev_machine/ 下的每个包链接或复制进来
cp -r /path/to/VLFM_GO2_STACK/dev_machine/base_autonomy/* .
cp -r /path/to/VLFM_GO2_STACK/dev_machine/slam/* .
cp -r /path/to/VLFM_GO2_STACK/dev_machine/unitree_webrtc_ros .
cp -r /path/to/VLFM_GO2_STACK/dev_machine/utilities/serial .
# vlfm_bridge / wireless_tools 不需要 colcon build，是独立脚本，见下面第6步

# 2) 雷达驱动（开发机需要它的消息类型定义才能跟 Jetson 的 /livox/lidar 互通；
#    Jetson 上同样需要一份，用来真正跑驱动节点）
git clone https://github.com/Livox-SDK/livox_ros_driver2.git
cd livox_ros_driver2/Livox-SDK2 2>/dev/null || git clone https://github.com/Livox-SDK/Livox-SDK2.git
# 按livox_ros_driver2自己的README编译安装Livox-SDK2，再回到 GO2_STACK_dev_ws colcon build

# 3) SLAM 依赖库（系统级安装，不进工作区）：Sophus / Ceres Solver / gtsam
#    分别 clone 官方仓库，按各自 README cmake && make && sudo make install 即可，
#    版本要求见 dev_machine/slam/arise_slam_mid360/package.xml

# 4) WebRTC 控制库（pip 装，不进 colcon 工作区）
python3 -m venv ~/unitree_venv && source ~/unitree_venv/bin/activate
pip install git+https://github.com/VectorRobotics/unitree_webrtc_connect.git

cd ~/GO2_STACK_dev_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Jetson 上另外还要 clone `livox_ros_driver2` + `Livox-SDK2` 一份（跑真正的驱动节点），以及官方 `autonomy_stack_go2`（`jizhang-cmu/autonomy_stack_go2`，本仓库不含，只用到它的 `ros_env_mid360.sh` 环境变量脚本）。详见 [`jetson/README.md`](jetson/README.md)。

## 1. 开始之前

- **人手里必须一直拿着实体遥控器**，随时能物理接管。这是唯一能取代的最后一道防线，软件安全设计都替代不了。
- 网络：开发机有线口接机器人，网段 `192.168.123.x`。
  - Jetson（机载电脑）`192.168.123.18`。
  - 机器人本体 `192.168.123.161`。
  - 开始前先 `ping` 一下两个地址；有线口如果显示 `NO-CARRIER`（`ip -br link`）就是网线/供电问题，先解决再往下。
- **换电池 / 机器人重启后，Jetson 也会一起重启**，Jetson 上的雷达驱动和相机推流全部没了，必须重来（见第 5 节"什么时候要重启什么"）。
- 每条 ROS 命令前先：
  ```bash
  source /opt/ros/jazzy/setup.bash
  source ~/GO2_STACK_dev_ws/install/setup.bash
  ```
  Shell 里设 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`，`ros2` 命令直接沿用。

**拉起顺序**（必须按这个顺序，前一步确认正常再做下一步）：

1. Jetson：雷达驱动 → 相机推流
2. 开发机：SLAM → 等约 14 秒 → sensor_scan_generation → terrain_analysis + terrain_analysis_ext → local_planner
3. 开发机：WebRTC（`unitree_control`）→ 验证 `/hello`
4. 开发机：VLM 服务（BLIP-2 + YOLO-World）
5. 健康检查 → 确认无误后再跑主脚本

其中 pathFollower、pose/waypoint/cmdvel 三个 UDP 中继**不需要手动启动**，由 `run_vlfm_pipeline.py` 每次运行时自己拉起、结束时自己关掉。

## 2. 第一步：Jetson 的雷达和相机

```bash
# 雷达驱动 —— 必须是 msg_MID360_launch.py（CustomMsg），不是 msg_MID360_pointcloud2_launch.py
ssh unitree@192.168.123.18 "source /opt/ros/foxy/setup.bash && source ~/livox/ws_livox/install/setup.bash && source ~/autonomy_stack_go2/ros_env_mid360.sh && setsid nohup ros2 launch ~/livox/ws_livox/src/livox_ros_driver2/launch_ROS2/msg_MID360_launch.py > /tmp/mid360_driver.log 2>&1 < /dev/null & disown; sleep 4; echo lidar launched"

# 相机推流（本仓库 jetson/realsense_stream_server.py，TCP 6000，支持多 client）
ssh unitree@192.168.123.18 "cd ~ && setsid nohup python3 realsense_stream_server.py > /tmp/realsense_stream_server.log 2>&1 < /dev/null & disown; sleep 6; echo realsense launched"
```

验证（开发机上）：
```bash
timeout -s INT 6 ros2 topic hz /livox/lidar      # 期望 ≈ 10 Hz
timeout -s INT 6 ros2 topic hz /livox/imu        # 期望 ≈ 200 Hz
(echo > /dev/tcp/192.168.123.18/6000) && echo camera OK
```

**Jetson 上 `zenoh-bridge-dds` 必须保持关闭**：它开机自启、占约 20% CPU，开着的时候 SLAM 的 IMU 预积分节点每秒失败约 1 次（`failureDetected`），`/state_estimation` 掉到 28~34 Hz、位姿会跳。
```bash
ssh unitree@192.168.123.18 'systemctl is-enabled zenoh-bridge-dds; ps -eo args | grep -c "[z]enoh-bridge"'   # 期望: disabled / 0
# 若不是：ssh unitree@192.168.123.18 'sudo systemctl stop zenoh-bridge-dds && sudo systemctl disable zenoh-bridge-dds'
```
（开发机上如果也装了 zenoh-bridge-dds，同样要停用并禁用。）

## 3. 第二步：开发机的导航栈

**机器人此时必须完全静止**（SLAM 在启动时用当前姿态初始化）。所有输出都重定向到文件，不然出问题什么都看不到。

```bash
source /opt/ros/jazzy/setup.bash; source ~/GO2_STACK_dev_ws/install/setup.bash

# 3.1 SLAM（feature_extraction + laser_mapping + imu_preintegration）
nohup ros2 launch arise_slam_mid360 arize_slam.launch.py > /tmp/arise_slam.log 2>&1 & disown
sleep 14          # 等 SLAM 初始化完，再起下游

# 3.2 下游
nohup ros2 launch sensor_scan_generation sensor_scan_generation.launch > /tmp/sensor_scan_generation.log 2>&1 & disown
sleep 2
nohup ros2 launch terrain_analysis terrain_analysis.launch > /tmp/terrain_analysis.log 2>&1 & disown
nohup ros2 launch terrain_analysis_ext terrain_analysis_ext.launch checkTerrainConn:=true > /tmp/terrain_analysis_ext.log 2>&1 & disown
sleep 2

# 3.3 local_planner —— 必须 autonomyMode:=true, twoWayDrive:=false
LOG=~/localPlanner_dbg_$(date +%m%d).log
nohup ros2 run local_planner localPlanner --ros-args \
  --params-file ~/GO2_STACK_dev_ws/install/local_planner/share/local_planner/config/unitree/unitree_go2_slow.yaml \
  -p pathFolder:=$HOME/GO2_STACK_dev_ws/install/local_planner/share/local_planner/paths \
  -p vehicleLength:=0.6 -p vehicleWidth:=0.5 -p sensorOffsetX:=0.2 -p sensorOffsetY:=0.0 \
  -p twoWayDrive:=false -p laserVoxelSize:=0.05 -p terrainVoxelSize:=0.2 -p useTerrainAnalysis:=true \
  -p checkObstacle:=true -p checkRotObstacle:=false -p adjacentRange:=3.5 \
  -p obstacleHeightThre:=0.05 -p groundHeightThre:=0.05 -p costHeightThre1:=0.1 -p costHeightThre2:=0.05 \
  -p useCost:=false -p slowPathNumThre:=5 -p slowGroupNumThre:=1 -p pointPerPathThre:=2 \
  -p minRelZ:=-0.4 -p maxRelZ:=0.3 -p maxSpeed:=0.2 -p dirWeight:=0.02 -p dirThre:=90.0 \
  -p dirToVehicle:=false -p pathScale:=0.875 -p minPathScale:=0.675 -p pathScaleStep:=0.1 \
  -p pathScaleBySpeed:=true -p minPathRange:=0.8 -p pathRangeStep:=0.6 -p pathRangeBySpeed:=true \
  -p pathCropByGoal:=true -p autonomyMode:=true -p autonomySpeed:=0.2 \
  -p joyToSpeedDelay:=2.0 -p joyToCheckObstacleDelay:=5.0 \
  -p goalClearRange:=0.35 -p goalBehindRange:=0.35 -p freezeAng:=90.0 -p freezeTime:=0.0 \
  > "$LOG" 2>&1 & disown
```

要点：
- **`local_planner` 必须 `autonomyMode:=true`**：它的转向方向只有在这个模式下才会从 `/way_point` 目标计算，否则永远只能从 `/joy` 得到（一台没有手柄发布者的机器人上，`/joy` 永远收不到消息），机器人就只会直走不转向。这个值是安全的，真正决定机器人会不会动的是 `pathFollower`，而它只在主脚本运行时才存在。
- `twoWayDrive:=false`：禁止倒车（摄像头朝前装，倒走时上层看不到新画面）。

## 4. 第三步：WebRTC（unitree_control）和 VLM 服务

```bash
# 4.1 WebRTC —— 必须用装了 unitree_webrtc_connect 的虚拟环境，并 unset conda 变量
nohup bash -c "unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH; source /opt/ros/jazzy/setup.bash && source ~/unitree_webrtc_ws/install/setup.bash && source ~/unitree_venv/bin/activate && exec ros2 launch unitree_webrtc_ros unitree_control.launch.py" > ~/unitree_control_$(date +%m%d).log 2>&1 & disown
sleep 14
ros2 service call /hello std_srvs/srv/Trigger     # 让机器人做打招呼动作，验证命令真能送达
```

- 启动日志里会有两条**看起来吓人但可以忽略**的报错：`port=8081 ... Max retries exceeded`（8081 端口一直是关的）和 `Failed to receive SDP Answer`，之后仍会打印 "Successfully connected to robot"。**这个 "Successfully connected" 不能当作已连通**，只有 `/hello` 返回 `success=True` 才算。
- **重连后的第一次 `/hello` 常常失败**（返回 `success=False`），再调一次通常就好；连续 2~3 次都失败就杀掉重连，再不行只能重启机器人本体。
- `/hello` 会让机器人动一下，**要确认有人在场再调**。

```bash
# 4.2 VLM 服务：主脚本只用到 BLIP-2（打分, 12182）和 YOLO-World（检测, 12185）
cd path/to/vlfm            # clone + 打好补丁的官方 vlfm，见 vlfm_patch/README.md
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 python -m vlfm.vlm.blip2itm  --port 12182 &
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 python -m vlfm.vlm.yolo_world --port 12185 &
for p in 12182 12185; do timeout 1 bash -c "echo > /dev/tcp/localhost/$p" && echo "port $p up"; done
```

YOLO-World 是开放词汇检测器，类别用 `' . '` 分隔的 caption。**只写单个泛词有时检测不到**（例如只写 `plant .` 得 0 个，写 `plant . potted plant . houseplant .` 才检出）。新增目标物时，在 `run_vlfm_pipeline.py` 的 `DET_SYNONYMS` 里加同义词。

## 5. 第四步：健康检查（跑主脚本之前必做）

| 检查项 | 命令 | 期望值 |
|---|---|---|
| 雷达 | `timeout -s INT 6 ros2 topic hz /livox/lidar` | ≈ 10 Hz |
| IMU | 同上 `/livox/imu` | ≈ 200 Hz |
| **位姿** | 同上 `/state_estimation` | **≈ 50 Hz**（28~34 Hz 说明 SLAM 在频繁失败） |
| 点云 | `/registered_scan`、`/terrain_map` | ≈ 3.3 Hz（设计值，不是问题） |
| 位姿数值 | `ros2 topic echo --once /state_estimation --field pose.pose.position` 连采几次 | 静止时稳定在毫米~厘米级，且离原点不远（几米以上说明该重启 SLAM） |
| SLAM 失败计数 | `grep -c failureDetected /tmp/arise_slam.log`，隔 10 秒再数一次 | 增量应为 0 |
| 相机 | 连 6000 端口取图 | 内参 fx≈390.9、640×480，多次取图亮度有细微变化（不是冻结） |
| WebRTC | `ros2 service call /hello std_srvs/srv/Trigger` | `success=True` |
| VLM | 12182、12185 端口 | 都开 |
| pathFollower | `ps -eo args \| grep "lib/local_planner/pathFollower"` | 应为空（由主脚本拉起） |

## 6. 第五步：运行主脚本

```bash
cd path/to/vlfm
PYTHONPATH=$(pwd) python \
    path/to/VLFM_GO2_STACK/dev_machine/vlfm_bridge/scripts/run_vlfm_pipeline.py \
    --target "trash can" --max-seconds 300 --record
```

- 会先要求输入 `ARM` 才真正开始运动。运行中输入 `target <名字>` 换目标，`stop` 或 `Ctrl-C` 安全停止（先杀 pathFollower，再 best-effort `/liedown`）。
- 运行流程：起中继 → 初始 360° 闭环旋转扫描（8 步×45°，此时还没有 pathFollower，由 `cmdvel_udp_relay` 只发偏航角速度）→ 把目标钉在当前位置、启动 pathFollower（`maxSpeed 0.2`、`twoWayDrive false`）→ VLFM 探索；连续 5 帧内有 3 帧检测到目标就锁定世界坐标，之后不再做上层推理；距离 ≤ 0.25 m 判定 FOUND，停下并趴下。
- 加 `--record` 会录第一人称视角视频，并在结束后自动合成带地图的复盘视频（`make_demo_video.py` / `make_composite_video.py`）。

无线组网模式下，`dev_machine/wireless_tools/` 里有对应的一键拉起（`bringup_nav_stack.sh`）、健康检查（`health_check.sh`）、关闭（`teardown_nav_stack.sh`）脚本，逻辑跟上面手动步骤一致。

## 7. 什么时候要重启什么

| 情况 | 要做的 |
|---|---|
| **换电池 / 机器人重启**（Jetson 也会重启） | 重做第 2 节（Jetson 雷达+相机）→ 重启整条 SLAM 链（杀掉 SLAM+下游，按第 3 节重起）→ 重连 WebRTC。 |
| 位姿离原点很远 / 位姿在跳 / `/state_estimation` 低于 50 Hz | 让机器人完全静止后重启 SLAM 链（不用重启雷达）；先查 Jetson 上 zenoh 是否又起来了。 |
| 相机画面冻结（彩色多次取图完全一样，深度仍在变） | Jetson 上杀掉 `realsense_stream_server.py` 重起，不用重启整机。 |
| WebRTC 数据通道断了（`/hello` 无响应/失败） | 杀 `unitree_control` 重起，再 `/hello`（第一次失败就再试一次）。 |
| 网线断了 / 有线口 `NO-CARRIER` | 恢复后所有依赖网络的部分都要重新检查（WebRTC、相机、雷达）。 |
| 只想换目标物 | 不用重启任何东西，主脚本里输入 `target <名字>`；新目标记得配好 `DET_SYNONYMS`。 |

## 8. 常见问题速查

| 现象 | 原因 / 处理 |
|---|---|
| `/state_estimation` 只有 28~34 Hz，日志里大量 `failureDetected` | 根因是 Jetson 上的 `zenoh-bridge-dds`。确认它已停并禁用。 |
| 位姿突然跳几米 | 同上；另外 SLAM 长时间运行后可能发散，重启 SLAM 链。 |
| 目标算出来了机器人不转向、只会直走 | local_planner 没带 `autonomyMode:=true`。 |
| 机器人倒着走 | `twoWayDrive` 不是 `false`（local_planner 和 pathFollower 都要）。 |
| YOLO-World 返回一堆 COCO 类别而不是要找的物体 | 首次 `set_classes` 的竞争问题，`vlfm_patch/yolo_world.py` 里已加锁和预热修复；仍出现就重启检测服务。 |
| `ModuleNotFoundError: rclpy._rclpy_pybind11` | shell 里 conda 是激活的，`unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH`。 |
| `unitree_control` 缺 `unitree_webrtc_connect` | 没进装了这个库的虚拟环境。 |
| 用 `pkill -f 模式` 后命令莫名退出 | `pkill -f` 会匹配到 shell 自己命令行里的同名字符串；改用 `ps -eo pid,args \| grep ... \| grep -v grep \| awk '{print $1}' \| xargs -r kill -9`。 |

## 9. 关闭流程

顺序很重要：**先在 WebRTC 还连着的时候让机器人趴下，再关其它**。

```bash
# 9.1 让机器人趴下，等确认成功再往下（要有人在场）
ros2 service call /liedown std_srvs/srv/Trigger

# 9.2 开发机上关闭（按 PID 杀，避免 pkill -f 误伤自己）
PAT="arise_slam_mid360|sensor_scan_generation|terrain_analysis|local_planner|unitree_control|vlfm\.vlm|run_vlfm_pipeline|pose_udp_relay|waypoint_udp_relay|cmdvel_udp_relay"
ps -eo pid,args | grep -E "lib/local_planner/pathFollower|cmdvel_udp_relay" | grep -v grep | awk '{print $1}' | xargs -r kill -9   # 先断 /cmd_vel 源头
ps -eo pid,args | grep -E "$PAT" | grep -v -E "grep|bash -c" | awk '{print $1}' | xargs -r kill -9

# 9.3 Jetson 上关闭雷达驱动和相机推流
ssh unitree@192.168.123.18 'ps -eo pid,args | grep -E "livox_ros_driver2|msg_MID360_launch|realsense_stream_server" | grep -v -E "grep|bash -c" | awk "{print \$1}" | xargs -r kill -9'
```

## 10. 已知、尚未彻底解决的问题

1. **WebRTC 会不定期静默断连**，没有根治，只有重连流程。
2. **位姿跳变**：停掉 Jetson 上的 zenoh 后，静止时 `failureDetected` 降到 0、`/state_estimation` 恢复 50 Hz，但运动（转圈/行走）时是否还会跳，还没在完整运行里验证。可用 `dev_machine/vlfm_bridge/scripts/analyze_pose_jumps.py` 分析 `ros2 bag record` 录下的位姿相关话题。
3. **Frontier 选择不考虑距离/方向**（vlfm 的 `value_map.sort_waypoints()` 只按语义分数排），绕场一圈后可能选中身后的旧高分点，需要大角度转向。
4. **到达阈值 0.25 m**：如果目标物本身被 local_planner 当作障碍，可能进不到 0.25 m；若距离卡住不降，可适当提高到 0.35~0.4 m。
