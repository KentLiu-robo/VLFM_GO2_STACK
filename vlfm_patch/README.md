# vlfm 补丁

对官方 [VLFM](https://github.com/bdaiinstitute/vlfm)（Boston Dynamics AI Institute）仓库的修改。**这里不是完整的 vlfm 仓库**，只放我们新增/改动的几个文件，其余代码请自己 clone 官方仓库。

## 怎么用

```bash
git clone https://github.com/bdaiinstitute/vlfm.git
cd vlfm
git checkout 584ed56008754fde7997d904983607def8328322   # 本项目实际基于的版本（origin/main, 2025-01-07）

# 按 vlfm 官方 README 装好环境和依赖（conda env, python3.9, habitat-sim 等）
# 装好后应用本仓库的改动：
cp /path/to/VLFM_GO2_STACK/vlfm_patch/yolo_world.py     vlfm/vlm/yolo_world.py
cp /path/to/VLFM_GO2_STACK/vlfm_patch/go2_robot.py       vlfm/reality/robots/go2_robot.py
cp /path/to/VLFM_GO2_STACK/vlfm_patch/obstacle_map.py    vlfm/mapping/obstacle_map.py

pip install --no-deps ultralytics   # YOLO-World需要，--no-deps避免动到已装好的numpy/torch/opencv版本
```

## 三个文件分别改了什么

### `vlm/yolo_world.py`（新增文件）

新增的开放词汇检测器，替换掉原来闭集的 YOLOv7（只认 COCO 80 类，`fan`/`trash can` 这类目标检测不到）。跟 GroundingDINO 一样用 `' . '` 分隔的 caption 格式。基于 [ultralytics](https://github.com/ultralytics/ultralytics) 的 YOLO-World 封装（`YOLOWorld` 本地推理类 + `YOLOWorldClient`/Flask server，跟 vlfm 其它 VLM 一致的 client/server 模式）。

两个需要注意的坑，代码注释里有详细说明：
- ultralytics 把 numpy 输入当 BGR（OpenCV 习惯），但调用方传的是 RGB——不翻转会让红蓝通道对调，实测置信度明显下降。
- Flask 每个请求起一个线程，而 client 的读超时（1s）比首次 `set_classes()`（触发 CLIP 文本编码，约 3s）短，会导致重试请求和还在跑的第一个请求同时改同一个模型对象（复现过：之后所有请求都返回默认 80 类 COCO 标签，不管你传了什么 caption）。用一把锁把 `set_classes` + `predict` 串行化解决。

### `reality/robots/go2_robot.py`（新增文件）

早期尝试用 vlfm 自带的机器人抽象接口（`reality/robots/`）直接对接 GO2 的一版实现，供参考。**实际跑的架构不是这条路**，最终走的是 `dev_machine/vlfm_bridge/scripts/run_vlfm_pipeline.py` 里更薄的一层（直接连相机 TCP + UDP 中继，不经过这套 robot 抽象类）。保留在这里是因为它包含了一部分传感器坐标变换相关的推导，可能有参考价值。

### `mapping/obstacle_map.py`（修改文件，两处 PATCH）

修的是 vlfm 自带的 `frontier_exploration` 库在真实硬件上暴露出的两个 bug（在其训练/评测用的 habitat-sim 房间里不会触发）。完整推导写在文件里对应位置的 `PATCH (2026-09-17)` 注释块中，这里只给结论：

1. **视野完全通畅时探索区域永久停止增长**：`reveal_fog_of_war` 的实现要求视锥内至少有一个障碍物轮廓才能算出有界的可见多边形，视野里没有障碍物时会直接返回空（"这个视锥里啥都没探索到"）。这个假设在训练用的房间里成立（总有近处的墙），但真实办公室的开阔走廊里不成立——实测：机器人视野一转到空旷走廊，`explored_area` 连续 48+ 个 tick 增量为 0，即使一直在走、也在持续正确检测到新障碍物。修法：视野确实干净（视锥内无障碍物）时，直接把这个视锥跟已知可通行区域的交集标记为已探索。
2. **单帧深度噪声导致已探索区域永久性坍缩**：真实深度传感器的噪声会在 `_navigable_map` 里打出小洞，使 `reveal_fog_of_war` 的射线提前中断，把本该连续的探索区域碎成好几个小岛；原本"只保留离机器人最近的一个轮廓，丢弃其它噪声碎片"的剪枝逻辑，如果那一帧运气不好导致机器人自己所在的像素恰好不在任何碎片里（不是真的丢失了历史探索记录，只是那一帧的碎片化让它找不到），就会把整块已经探索的区域当噪声全部丢弃——实测：一次坏帧把 1474 像素的已探索区域砸到 1019，之后连续 48+ 个 tick 卡在这个坍缩后的大小不再变化。
