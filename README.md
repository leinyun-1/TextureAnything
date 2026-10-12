# TextureAnything

训练一个视频生成模型：给真实拍摄的弱纹理物体视频加上一层贴在表面上的纹理，使 COLMAP 对这个物体的重建更好。当前阶段只做数据。模型训练要等这批 pair 在 GPU 服务器上渲完之后再开始。

## 任务

推理时先把物体从真实视频里抠出来，背景填黑，再交给生成模型和 COLMAP。因此训练数据按同一个界面制作：

- 只渲染物体，背景纯黑。
- 光照来自实拍 HDRI，接近真实受光。相机射线看到的是黑色，环境光只照在物体上。
- 每对视频的相机不固定。焦距、距离、弧长、仰角和起始角按物体 id 抽样，同一对的两段视频共用这条轨迹和同一张 HDRI。
- 一段是原始反照率，一段是在物体表面上叠加跨视角稳定的斑点。只有物体表面不同，相机和光照相同。

黑背景和物体交界会形成一圈假轮廓。COLMAP 评测时用 mask 把特征限制在物体内部。弱纹理物体在纯黑背景下往往无法注册；加上跨视角稳定的纹理之后应当能注册。这是这批数据要服务的检验。

真实抠图仍带着房间的光照、曝光和边缘误差。HDRI 负责让物体上的明暗接近实拍，不负责把房间画进画面。

## 一对数据

队列里有 1673 个 ABO 物体。每个物体一个目录：

```
renders_black/<id>/pair.json
renders_black/<id>/P0_plain/
renders_black/<id>/P4_speckle/
```

| 目录 | 内容 |
| --- | --- |
| `P0_plain` | 原始反照率，粗糙度 0.88，金属度 0 |
| `P4_speckle` | 同一网格，反照率改为 `clip(A + 0.4 F, 0, 1)`，`F` 是物体空间 FBM |

固定参数，不要改：

- 宽 832，高 480
- 每段 81 帧，`rgb/0000.png` 到 `rgb/0080.png`
- Cycles，32 samples，OpenImageDenoise
- 斑点 scale 为 medium（24）
- 物体最大边归一化到 1，摆在原点
- 焦距约 34–52 mm，物体约占画面高度的 50%–66%，弧长 140°–300°，仰角和起始角每对不同
- 光照从 20 张 Poly Haven CC0 室内 HDRI 中按物体 id 选取，并随机绕竖直轴旋转

一对算完成：两个目录都有 `rgb/0080.png`，且 `camera_gt.json` 中 `width=832`、`height=480`、`frames` 长度为 81。

每帧同时写出 RGB、mask 和深度。mask 用于把 COLMAP 特征限制在物体内部。深度供以后和真值对比，体积很大，约 400 GB。

## 在 GPU 服务器上渲染

本机是 Windows。服务器按 8 张 NVIDIA H20 的 Linux 机器准备。Windows 版 Blender 和本机 conda 环境不能拷过去运行。

服务器需要：

- 能识别 H20 的 NVIDIA 驱动
- Blender 5.2.2 Linux x64
- Python 3.10 或 3.11，安装 `pyyaml`、`numpy`、`opencv-python-headless`

代码在本仓库。数据不在 Git 里，需要单独放到服务器：

```
DATA_ROOT/abo/p0_queue.json
DATA_ROOT/abo/models/<物体id>.glb
DATA_ROOT/hdri/<名字>_2k.hdr
```

制作机器上的对应位置：

- 队列：`D:\TextureAnything\datasets\abo\p0_queue.json`
- 模型：`D:\TextureAnything\datasets\abo\models`，1673 个 glb，约 20.6 GB
- HDRI：`D:\TextureAnything\datasets\hdri`，20 张 `*_2k.hdr`，约 121 MB

队列 json 里的 `glb` 字段是 Windows 路径。不要改 json，用 `--models-dir` 按物体 id 找文件。不要把未完成的 `renders_black` 或室内场景拷到服务器。输出从空目录开始，并预留 600 GB 以上空间。

先用一张卡渲队列中的 1 个物体：

```bash
export DATA_ROOT=/path/to/data
export BLENDER=/path/to/blender-5.2.2-linux-x64/blender
CUDA_VISIBLE_DEVICES=0 python -u scripts/render_abo_black.py \
  --queue "$DATA_ROOT/abo/p0_queue.json" \
  --models-dir "$DATA_ROOT/abo/models" \
  --hdri-dir "$DATA_ROOT/hdri" \
  --render-root "$DATA_ROOT/renders_black" \
  --blender "$BLENDER" \
  --shard-index 0 --shard-count 8 --limit 1
```

确认第一帧是 832×480、四角接近纯黑、物体占画面高度大约一半以上，并且 P4 相对 P0 只在物体表面多出斑点。日志里应出现 `Cycles device=OPTIX`。OptiX 不可用时脚本会改用 CUDA。

通过后再把 1673 个物体分给 8 张卡。分片按队列顺序，`shard-index` 从 0 到 7，`shard-count` 为 8。每张卡单独一个进程，用 `CUDA_VISIBLE_DEVICES` 绑定。不要把 8 张卡绑进同一次 Blender 渲染，单帧只有大约两秒。不要改 `p0_queue.json` 的顺序。

`run_shards.sh` 在 `dist/gpu_render_task/`，作用和上面的 8 次调用相同。中断后用同一命令重跑，已完成的 pair 会跳过。

## 仓库内容

纳入版本库的是代码、配置和这份说明：

- `scripts/blender_render.py`：Blender 内的 Cycles 渲染。黑背景 HDRI、相机轨迹、P0/P4 材质都在这里。
- `scripts/render_abo_black.py`：ABO 队列调度，支持路径参数和按 GPU 分片。
- `scripts/render_dataset.py`：调用 Blender，并把 mask 写成 png。
- `scripts/config_utils.py`、`configs/`：渲染和图案配置。
- `requirements.txt`：调度脚本的 Python 依赖。

以下内容不进入版本库：

- `data/`、`results/`：已有实验的渲染、COLMAP 结果和清单。
- 模型、HDRI、图片、深度、网格和压缩包。
- `dist/`：给服务器打包出来的副本，源文件以仓库里的 `scripts/` 为准。

更早的灰棚渲染和室内场景实验留在制作机器上，不作为这批训练数据。灰棚结果只说明物体空间斑点本身能提高 COLMAP；室内场景后来改为不用，因为推理会先抠出物体。

## 现在不要做的事

- 不要改分辨率、帧数、黑背景、HDRI 光照和去噪设置。
- 不要开始训练视频模型。
- 不要在这批数据完成之前跑大规模 COLMAP。
