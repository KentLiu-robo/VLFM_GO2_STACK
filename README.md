# VLFM_GO2_STACK

Unitree GO2 四足机器人 + [VLFM](https://github.com/bdaiinstitute/vlfm) 的语义目标导航（ObjectNav）实机部署代码。给机器人一个文字目标（比如 `"trash can"`），它自主探索环境、用视觉语言模型判断朝哪走更可能找到目标，找到后自动停下。

整理于 2026-09-22。这不是一个可以直接 `colcon build` 的完整工作区快照，而是**本项目实际编写/修改过的代码**，按开发机（PC）、Jetson 机载电脑、vlfm 三个部分分开存放。跑起来还需要另外 clone 几个第三方仓库（雷达驱动、gtsam/ceres 等 SLAM 依赖库、VLFM 本体），具体见 [`QUICKSTART.md`](QUICKSTART.md)。

## 仓库结构

```
VLFM_GO2_STACK/
├── dev_machine/            # 开发机 (PC, ROS 2 Jazzy) 端代码
│   ├── base_autonomy/      # 底层避障导航: local_planner(含pathFollower) / terrain_analysis(+ext) / sensor_scan_generation
│   ├── slam/                # arise_slam_mid360 (SLAM) + 消息定义包
│   ├── unitree_webrtc_ros/  # 订阅 /cmd_vel，通过 WebRTC 直接控制机器人本体
│   ├── vlfm_bridge/         # ★ vlfm ↔ ROS2 的桥接层，见下方"这是什么"
│   ├── wireless_tools/      # 无线组网模式下的一键拉起/健康检查/关闭脚本
│   └── utilities/serial/    # local_planner 的编译期依赖（第三方小库，已带）
├── jetson/                  # Jetson 机载电脑端代码
│   └── realsense_stream_server.py   # 自研的相机 TCP 推流服务，见 jetson/README.md
└── vlfm_patch/               # 对官方 VLFM 仓库的修改（不是完整 vlfm 仓库），见 vlfm_patch/README.md
```

`dev_machine/vlfm_bridge` 是这个项目里**唯一从零编写**的部分，其余 `dev_machine/` 目录都是对 [`autonomy_stack_mecanum_wheel_platform`](https://github.com/jizhang-cmu/autonomy_stack_mecanum_wheel_platform)（Ji Zhang 组）底层导航算法的复用（未改动核心算法，只改了少量配置/接口，具体见各文件的内联注释）。**没有带**该上游仓库里跟当前 GO2+vlfm 流程无关的部分（模拟环境、route/exploration planner、Mecanum 轮特有的遥操作/串口电机驱动等）。

## 系统架构与信息流

### 0. 先明确三台"电脑"

```
开发机 (这台PC, ROS 2 Jazzy)  ──有线网口──  Jetson机载电脑 (192.168.123.18, ROS 2 Foxy)
        │                                          │
        │                                    (运行雷达驱动+相机推流)
        │
        └──WebRTC(同网段)──  GO2本体控制器 (192.168.123.161)
                              (运行 SPORT_CMD 执行层，机器人真正的"大脑")
```

这是**三台**独立计算单元，不是两台。Jetson 只负责雷达和相机这两个传感器的采集与转发，不做任何导航决策；真正让机器人动起来的指令，是从开发机直接通过 WebRTC 发给机器人本体的，**不经过 Jetson**。

### 1. Jetson → 开发机：两条完全独立的通路

**雷达 + IMU：原生 ROS 2 DDS，不是自定义中继**

Jetson 上 `livox_ros_driver2`（`msg_MID360_launch.py`，第三方仓库，本仓库不含，见 [`QUICKSTART.md`](QUICKSTART.md)）直接把 `/livox/lidar`（自定义 `CustomMsg`，10 Hz）和 `/livox/imu`（200 Hz）发布成标准 ROS 2 话题。开发机上的 SLAM 节点直接订阅——**中间没有任何网桥或中继脚本**，靠的是 DDS 本身跨网段发现（两边都设了 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`、`ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`）。Jetson 跑 ROS 2 Foxy、开发机跑 Jazzy，发行版不同但能互通，因为 DDS 的线缆协议（RTPS）本身不区分发行版，只要消息类型哈希对得上。

**`zenoh-bridge-dds` 必须保持关闭**（Jetson 和开发机两侧都要停并禁用）。它曾一度自启动并占约 20% CPU，扰乱雷达/IMU 数据到达的节奏，导致 SLAM 的 IMU 预积分节点频繁失败（`underconstrained call to isam2` → `failureDetected`）、`/state_estimation` 从正常 ~50 Hz 掉到 28~34 Hz、位姿出现跳变。

**相机：完全不走 ROS，是自定义 TCP 协议**

`jetson/realsense_stream_server.py` 在 Jetson 上开一个 **TCP 6000** 端口，用自定义二进制协议直接发流。这是刻意设计，不是偷懒：vlfm 用的 conda Python（3.9）环境里 `rclpy` 的编译版本跟系统 ROS（Jazzy，Python 3.12）不兼容，vlfm 进程没法正常订阅 ROS 话题，所以相机数据干脆不进 ROS。

协议细节（对应 `dev_machine/vlfm_bridge/scripts/run_vlfm_pipeline.py` 里的 `_RealSenseStreamClient`）：
- 连上先收一个固定 24 字节的内参包：`>4fII f`（fx, fy, ppx, ppy, width, height, depth_scale，大端序）。
- 之后循环：先收 8 字节头 `>II`（彩色帧字节数、深度帧字节数），再收对应长度的 JPEG/PNG 编码字节，`cv2.imdecode` 解出彩色图和深度图。
- 支持多个 client 同时连。

### 2. 开发机内部：ROS 2 导航栈（原生话题，全部 Jazzy）

```
/livox/lidar, /livox/imu
        │
        ▼
arise_slam_mid360 (feature_extraction_node → laser_mapping_node → imu_preintegration_node)
        │
        ├─→ /laser_odometry, /aft_mapped_to_init_incremental  (雷达建图的原始位姿)
        └─→ /state_estimation  (融合IMU后的最终位姿, 正常~50Hz)
        │
        ▼
sensor_scan_generation  →  /registered_scan (~3.3Hz, 世界系点云)
        │
        ▼
terrain_analysis + terrain_analysis_ext  →  /terrain_map, /terrain_map_ext (~3.3Hz, 可通行性栅格)
        │
        ▼
local_planner (消费 /state_estimation + /terrain_map(+ext) + /way_point)
        │  只在 autonomyMode:=true 时才会用 /way_point 算转向方向
        ▼
        /path  (候选无碰撞路径)
        │
        ▼
pathFollower (local_planner 包里的另一个可执行文件，消费 /path + /state_estimation)
        │  只在自己的 autonomyMode 打开、且路径点数>1 时才会真的发非零速度
        ▼
        /cmd_vel  (TwistStamped)
```

`/way_point` 是唯一从"上层语义决策"（vlfm）注入进来的输入，其余全是导航栈自己的闭环。

### 3. vlfm ↔ ROS 2 导航栈的桥接：`dev_machine/vlfm_bridge`，三个本地 UDP 中继

这是整套系统里最不"标准"的部分，存在的**唯一原因**是 vlfm 的 conda 环境里 `rclpy` 用不了。解决办法是：用系统自带的 ROS Python 单独起三个小节点，它们能正常收发 ROS 话题，vlfm 进程通过 `127.0.0.1` 上的 UDP 跟这三个节点对话，由它们代为发布/订阅。

| 中继 | 端口 | 方向 | 包格式 | 作用 |
|---|---|---|---|---|
| `pose_udp_relay.py` | 8765 | ROS→vlfm | `<8d`: stamp, x, y, z, qx, qy, qz, qw | 订阅 `/state_estimation`，把位姿转发给 vlfm，供它给自己的 ObstacleMap/ValueMap 定位 |
| `waypoint_udp_relay.py` | 8766 | vlfm→ROS | `<3d`: x, y, z | vlfm 选好探索目标点后发过来，由它发布成 `/way_point`（`PointStamped`）给 local_planner |
| `cmdvel_udp_relay.py` | 8767 | vlfm→ROS | `<d`: wz（仅偏航角速度） | **只在初始 360° 扫描阶段使用**，此时 pathFollower 还没启动，由它直接发布 `/cmd_vel`，线速度硬编码为 0，`|wz|` 限幅 0.6 rad/s，指令流一停就发零速并静默 |

三个中继都是每次运行 `run_vlfm_pipeline.py` 时脚本自己拉起、结束时自己杀掉，不需要手动常驻。`cmdvel_udp_relay` 和 `pathFollower` 从不同时向 `/cmd_vel` 发布，所以不会互相打架。

### 4. vlfm 进程内部：决策怎么产生的

```
相机(TCP 6000)彩色图+深度图+内参  ──┐
位姿(UDP 8765)                     ├─→ ObstacleMap / ValueMap 更新(vlfm自己的几何+语义建图)
                                    │
                                    ├─→ 彩色图送本地HTTP:
                                    │     BLIP2ITM  (localhost:12182, Flask) → 语义匹配分数
                                    │     YOLO-World (localhost:12185, Flask) → 目标检测框
                                    │
                                    ▼
                            frontier 打分 → 选出下一个探索点(或已连续多帧检测到目标→锁定目标世界坐标)
                                    │
                                    ▼
                        waypoint_udp_relay(UDP 8766) → /way_point
```

BLIP-2 和 YOLO-World 也在本机内，走最普通的 HTTP POST + JSON（vlfm 自带的 `server_wrapper.py`），跟 ROS、跟 Jetson 都没关系。YOLO-World 检测器（`vlfm_patch/yolo_world.py`）是本项目新增的，用它替换了原本闭集的 YOLOv7，见 [`vlfm_patch/README.md`](vlfm_patch/README.md)。

### 5. 开发机 → GO2 本体：WebRTC，直接对机器人，不经过 Jetson

```
/cmd_vel (pathFollower 或 cmdvel_udp_relay 发布)
        │
        ▼
unitree_control (dev_machine/unitree_webrtc_ros 节点)
        │  订阅 /cmd_vel，转换成 SPORT_CMD["Move"]
        ▼
WebRTC 数据通道 (LocalSTA 模式, 直连机器人本体 192.168.123.161)
        │
        ▼
GO2 本体运动控制器执行
```

这条腿是开发机**直接**连机器人本体的 IP（`.161`），跟 Jetson（`.18`）是平行的两条网络路径，不是"PC→Jetson→机器人"这种串联关系。WebRTC 握手细节尚未完全查清：启动日志里两种握手尝试（HTTP POST 到机器人 8081、旧版 SDP 协商）都会报错，但连接最终仍能建立并可用——只能靠 `ros2 service call /hello std_srvs/srv/Trigger` 是否返回 `success=True` 来判断连接是否真的可用。

### 6. 为什么要设计成这样（一句话总结）

**唯一的根本原因**是 vlfm 用的 conda Python 3.9 环境里编译的 `rclpy` 跟系统 ROS 2 Jazzy（Python 3.12）ABI 不兼容，vlfm 进程没法直接做 ROS 的发布/订阅。由此派生出两个选择：
- 需要和 ROS 话题打交道的部分（位姿、目标点、扫描阶段的转向指令）→ 用系统 Python 单独起三个"传声筒"节点，vlfm 进程用本地 UDP 跟它们说话（`dev_machine/vlfm_bridge`）。
- 不需要经过 ROS 的部分（相机数据、VLM 推理）→ 干脆绕开 ROS，直接用普通的 TCP/HTTP，顺便还换来了相机支持多 client 连接这个额外好处。

## 完整拉起 / 运行步骤

见 [`QUICKSTART.md`](QUICKSTART.md)。

## Credits

底层导航栈（`dev_machine/base_autonomy`、`dev_machine/slam`）fork 自 [Ji Zhang's](https://frc.ri.cmu.edu/~zhangji) 团队（Carnegie Mellon University）的 [`autonomy_stack_mecanum_wheel_platform`](https://github.com/jizhang-cmu/autonomy_stack_mecanum_wheel_platform) / [`autonomy_stack_go2`](https://github.com/jizhang-cmu/autonomy_stack_go2)。SLAM 模块是 [LOAM](https://github.com/cuitaixiang/LOAM_NOTED) 的升级实现，底层避障导航基于 [Autonomous Exploration Development Environment](https://www.cmu-exploration.com)。

语义目标导航基于 [VLFM](https://github.com/bdaiinstitute/vlfm)（Boston Dynamics AI Institute）的 ObstacleMap/ValueMap 前沿探索方法，检测器为 [YOLO-World](https://github.com/AILab-CVC/YOLO-World)（经 [ultralytics](https://github.com/ultralytics/ultralytics) 封装），语义打分为 BLIP-2 ITM。

雷达驱动 [livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2) / [Livox-SDK2](https://github.com/Livox-SDK/Livox-SDK2)、WebRTC 通信库 [unitree_webrtc_connect](https://github.com/VectorRobotics/unitree_webrtc_connect)、SLAM 依赖 [gtsam](https://gtsam.org) / [Ceres Solver](http://ceres-solver.org) / [Sophus](http://github.com/strasdat/Sophus.git)、`dev_machine/utilities/serial` 来自 [wjwwood/serial](https://github.com/wjwwood/serial)，均为第三方开源项目，本仓库均未改动核心代码。
