# Jetson 机载电脑端

Jetson（`192.168.123.18`，ROS 2 Foxy）上运行两件事：雷达驱动、相机推流。它不做任何导航决策，只是传感器数据的采集/转发；真正让机器人动的指令是开发机通过 WebRTC 直接发给机器人本体的（不经过 Jetson）。见仓库根目录 README 的"系统架构与信息流"一节。

## `realsense_stream_server.py`（本仓库唯一带的 Jetson 端代码，自己写的）

RealSense D435i 的彩色+深度流通过一个自定义 TCP 服务（端口 6000）推给开发机，不走 ROS——因为开发机上跑 vlfm 的 conda Python 3.9 环境里 `rclpy` 跟系统 ROS（Jazzy, Python 3.12）不兼容，没法正常订阅 ROS 话题。协议格式见文件内 docstring 和仓库根目录 README 第 1 节。

支持多个 TCP client 同时连接（方便一边跑管线一边单独连上去做健康检查）。

启动：
```bash
python3 realsense_stream_server.py
```

**⚠️ 这份文件是 2026-09-19 的本地备份快照，不保证是 Jetson 上当前运行的最新版本。** 后续至少有一次改动没有同步进来：给推流速率加了上限（环境变量 `RS_MAX_FPS`，0 = 不限速）。用之前建议先 `diff` 一下 Jetson 上实际在跑的文件，确认没有更晚的改动被漏掉。

## 没有带的部分（第三方 vendor 代码，不是本项目编写的）

- **雷达驱动**：[`livox_ros_driver2`](https://github.com/Livox-SDK/livox_ros_driver2) + [`Livox-SDK2`](https://github.com/Livox-SDK/Livox-SDK2)，官方仓库直接 clone 编译即可，本仓库未修改其代码，只改了一处配置：`config/MID360_config.json` 里的雷达 IP（改成 `192.168.1.1xx`，xx 是雷达序列号后两位，具体值因设备而异，装机时自己填）。
  启动方式：`msg_MID360_launch.py`（不是 `msg_MID360_pointcloud2_launch.py`——后者发 PointCloud2，是给别的架构用的；`arise_slam_mid360` 需要雷达原生的 `CustomMsg`）。
- **`~/autonomy_stack_go2/ros_env_mid360.sh`**：来自 [`jizhang-cmu/autonomy_stack_go2`](https://github.com/jizhang-cmu/autonomy_stack_go2)，只用到它的环境变量设置脚本（设 `ROS_DOMAIN_ID`/`RMW_IMPLEMENTATION` 等），本项目没有修改。如果不想 clone 整个仓库，直接手动 `source /opt/ros/foxy/setup.bash` 后按需设置这几个环境变量也可以。
