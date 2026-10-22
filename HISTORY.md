# 项目历史

按时间记录做过的实验、得到的结论，以及每批数据放在哪里。任务定义和运行方法在 `README.md`。

数据都不在 Git 里。下文的 `data\`、`results\` 指项目根目录下被忽略的目录，其余是制作机器上的绝对路径。

## 时间线

### 2026-09-24 至 09-28：合成基本体上的纹理对照

预研方案见 `纹理增强预研方案.md`。三个基本体（box、bottle、bowl），基础反照率是均匀灰色，24 帧环绕，Cycles 渲染。比较 7 种表面图案：P0 纯灰、P1 逐帧噪声、P2 周期网格、P3 随机点、P4 物体空间稠密散斑、P5 多尺度散斑、P6 位置哈希。

- SIFT 正确匹配：P4 约 83，P0 约 0.3，P1 约 0。P1 特征点最多，但跨帧不一致，几乎无法匹配。
- SfM：P0 和 P1 在三个物体上都是 0/24 注册，P4 都是 24/24，旋转误差中位数 0.16°–1.37°。
- MVS（所有组共用真值位姿）：P4 的 Chamfer 约为 P0 的 1/6，P1 几乎融合不出点。

结论：起作用的是跨视角一致、贴在表面上的纹理，图像花不花本身无关。

局限：均匀灰模在现实中没有对应物，这组结果只说明“给完全没有特征的表面加特征，SIFT 能用上”，不能代表真实失败。

位置：`data\smoke`、`results\smoke\report.md`、`results\smoke\sfm_metrics.csv`、`results\smoke\dense_metrics.csv`。

### 2026-09-28 至 09-29：GSO 真实贴图上的 P4

从 Google Scanned Objects 选弱纹理物体，P0 用原扫描贴图，P4 在原贴图上叠加 `A' = clip(A + 0.4F, 0, 1)`。

- 第一批 5 个白瓷（mug、serving_bowl、plate、pitcher、teapot）：注册率 0.11 → 0.69；GT 位姿下 MVS 的 Chamfer 下降约 45%。
- 自动筛选的 24 个新物体：注册率 0.16 → 0.73，留出的 8 个是 0.16 → 0.63。P0 有 13 个完全失败，P4 没有失败。盘子这类近平面物体提升小。
- 筛选脚本 `scripts/select_gso_weak.py`：先按名字做关键词过滤，再对贴图的梯度、饱和度和亮度标准差打分。

位置：`data\gso_assets`（模型和贴图）、`data\gso_p4`、`data\gso_scale`、`data\gso_select`、`results\gso_p4`、`results\gso_scale`。

### 2026-09-30：用投影灯打出散斑，替代修改反照率

推理时改不了物体反照率，所以试了用带散斑图案的聚光灯照到物体上（P4P）。

- 3 盏灯：在 5 个 P0 全部失败的难例上，注册率 0 → 0.26，只保住约 8% 的 oracle 增益。
- 8 盏灯：注册率 0 → 0.45，保住约 20% 的增益。修改反照率的 P4 注册率是 0.67。

结论：打光比不打好，但远弱于修改反照率，没有作为数据制作方式。

位置：`data\gso_proj`、`data\gso_proj8`、`results\gso_proj`、`results\gso_proj8`。

### 2026-10-08 至 10-09：数据管线与 ABO

- GSO 上第二轮贴图筛选，153 个候选里只剩 2 个合格，库本身不够用。
- 下载 Amazon Berkeley Objects（ABO）三维模型，约 154 GB。按与 GSO 相同的贴图门槛给 7953 个模型打分，1792 个通过。最终队列 1673 个，见 `p0_queue.json`。
- 用户提出：生成模型只需要同轨迹的配对，原物体是不是弱纹理并不要求，强纹理物体也可以做配对。此后不再对每个物体跑 COLMAP 作门槛。
- 在灰色摄影棚里用 Cycles 渲完 1673 对 P0 和 P4，每段 24 帧、512×512，相机轨迹固定。这批后来没有使用。

管线代码：`scripts/run_pipeline.py`、`scripts/prepare_abo.py`、`scripts/screen_abo.py`、`scripts/render_abo_p0.py`、`scripts/render_abo_p4.py`。账本在 `data\pipeline`。

### 2026-10-10：室内场景

为了缩小和真实视频的差距，用 Infinigen 生成室内房间，把 ABO 物体放在桌面上渲染配对。生成了餐厅和卧室各若干间，出过相机穿进柜子、物体太小、噪点多的问题。

后来用户决定：推理时先用 mask 把物体抠出来，所以训练数据只渲染物体、背景纯黑，室内场景不再使用。

位置：`D:\TextureAnything\indoors`（场景、配对和脚本）、`scripts/render_indoor_pair.py`。

### 2026-10-11 至 10-13：纯黑背景与 HDRI 光照

- 背景纯黑，光照来自 20 张 Poly Haven 室内 HDRI，相机射线看到黑色。下载脚本是 `scripts/download_hdri.py`。
- 每个物体的焦距、弧长、仰角、起始角和 HDRI 由物体 id 决定，同一对共用。
- 分辨率定为 832×480、81 帧，只存 RGB 和 `camera_gt.json`。
- 渲染器改为 EEVEE（光栅化）。约 0.13 秒一帧，本机估计 12 小时能渲完 1673 对。
- 取景改为物体包围盒投影后较长的一边占画面高度的 80%–90%，并保证每一帧物体都完整留在画面内。
- 3 个物体的 SfM 测试中，散斑改善了重建，但画面上的散斑很淡：scale 24 的 FBM 在像素内被平均掉，亮色表面又把正向扰动截断了。

这一阶段建了 Git 仓库并推送到 GitHub（提交 `da45ef3`），给服务器准备了任务包 `dist\gpu_render_task`，还把数据打包成了 `E:\TextureAnything-data.zip`。1673 对的全量渲染没有执行。

位置：`D:\TextureAnything\datasets\abo\renders_test`（3 个物体的 EEVEE 测试，含 COLMAP 结果）、`D:\TextureAnything\datasets\abo\renders_black`（中途停掉的旧渲染，分辨率混杂，用的是 Cycles，已作废）。

### 2026-10-14 至 10-15：重新审视立意

讨论中提出的问题：

- 低纹理物体是否足够常见。
- 生成跨视角一致的纹理，等于模型已经掌握了对应关系；VGGT、MASt3R 这类方法直接输出相机和点图。
- 早期 oracle 都建立在灰模或强制哑光的表面上，没有对应的真实数据。

讨论出的方向：把视频改写成经典重建所假设的样子，也就是“数字扫描喷雾”。工业上给反光、透明零件喷显像剂，正是为了这个。重点从低纹理转到金属、亮面和透明物体。真实世界里的喷雾对照实验难度太大，没有做。用户暂停了推进。

### 2026-10-19 至 10-21：反光物体的 COLMAP 实验

4 个 ABO 物体：花瓶 B075HWK9QT、平底锅 B07BLB6644、双层床 B07BG69MRW（ABO 把它标成了 DRINKING_CUP）、置物架 B0754KHKCX。每个物体渲 7 种材质：`matte`、`glossy`、`abo_pbr`、`white_matte`、`white_glaze`、`chrome`、`brushed_metal`。EEVEE 打开屏幕空间光追，黑背景加 HDRI，832×480、81 帧，同一物体的所有材质共用相机和光照。

没有散斑时：

- 金属表面往往“看起来成功”：注册的帧数和点数都比哑光多，但位姿是错的。双层床镀铬 80/81 帧、旋转误差 86.8°；平底锅的 ABO 原始金属材质 81/81 帧、18.5°。
- 非金属亮面视物体而定：双层床白釉 0.59°，平底锅白釉 16.9°，置物架亮漆只注册 2 帧。
- 花瓶在 7 种材质下全部失败，包括哑光，因为它本身没有纹理。

加散斑后：散斑是印在表面上的哑光颜料点，黑点反射率 0.03，白点 0.85，尺度 28，阈值 0.08，覆盖约四成表面，其余部分保留原材质和反光。28 段视频全部 81/81 注册，旋转误差中位数都不超过 1.07°。

结论：生成模型不需要去掉反光，只要叠加一部分跟随表面的高对比度点。但这仍是用真值几何贴上去的 oracle。

局限：4 个物体、每个一条轨迹；EEVEE 不能准确计算物体自身的相互反射；镀铬双层床这类组合只能当作压力测试；带长柄的物体在画面里偏小，因为取景用的是包围盒角点。

代码：`scripts/exp_reflective_colmap.py`（渲染、COLMAP 和评测）、`scripts/make_reflect_videos.py`（对比视频）。`blender_render.py` 新增了 `--material`、`--raytrace`、`--speckle` 等参数。

位置：
- 渲染：`D:\TextureAnything\datasets\abo\renders_reflect\<物体>\<材质>` 和 `<材质>_speckle`。
- 对比视频：`D:\TextureAnything\datasets\abo\renders_reflect\videos`。
- 结果表：`results\reflect_colmap\sfm_metrics.csv`、`results\reflect_colmap\sfm_metrics_speckle.csv`。
- 调参测试图：`D:\TextureAnything\datasets\abo\probe_reflect`，未参与评测，可以删除。

### 2026-10-21：让其他机器能快速复现

- 物体队列去掉 Windows 绝对路径，放进仓库：`manifests/abo_queue.json`，保留原顺序，GPU 分片不变。制作机器上的 `p0_queue.json` 保留，但脚本不再读取。
- 数据根目录和工具路径改为可以用 `TA_DATA_ROOT`、`TA_BLENDER`、`TA_COLMAP` 覆盖，在 `scripts/config_utils.py` 里统一定义。
- `scripts/fetch_data.py` 按单个文件从 ABO 的 S3 下载队列里的 glb（约 20.6 GB），不再需要 154 GB 的整包，也负责下载 HDRI。`scripts/check_setup.py` 检查环境和数据。步骤写在 `SETUP.md`。
- 修复 `colmap_model.pick_largest_model`：COLMAP 输出多个子模型时，它原本拿 `images.txt` 里的图像数和 `images.bin` 的字节数直接比较，可能选中较小的子模型。反光实验的 56 个场景用修复后的代码重新评测，结果完全一致。早期 GSO 实验里有 7 个 P4 场景选错了子模型，都是选了注册帧更少的那个，比如 `gso_p4` 的 plate 实际是 13/24，报告里写的是 8/24。所以早期报告低估了 P4 的效果。这些结果表还没重新生成。

### 2026-10-22：数据集选择

训练配对要求「物体自己的材质」对「同一表面上加了散斑」。材质必须是数据里自带的金属度、粗糙度，不能再把网格改成镀铬。按这个标准看过现有来源和几个公开数据集。

| 数据集 | 规模 | 自带 PBR | 和本任务的关系 |
| --- | --- | --- | --- |
| TexVerse PBR | 15.8 万个 glb，带金属度或高光、粗糙度贴图，另有文字标注。全集 85.9 万 | 有 | 定为训练集来源。能筛出大量金属、陶瓷、玻璃，glb 直接接上现有 Blender 管线 |
| ABO（现有队列） | 1673 个 | 有 | 留作家具类弱纹理的补充。灯具 373、椅子 194、桌子 164、沙发 130；花瓶 21、平底锅 3。主体是带贴图的家具 |
| GSO、OmniObject3D | 约 1 千、6 千 | 无，只有漫反射贴图 | 扫描时高光被消掉了，金属物体本来就少 |
| DTC | 约 2400 个实物扫描，4K PBR。另有 50 个物体的 DSLR 环绕和 100 个物体的 Aria 录像 | 有 | 材质最接近实物，量不够单独训练。许可是 CC BY-NC 4.0。DSLR 序列适合做真实测试 |
| Stanford-ORB | 14 个物体、42 个场景，HDR 照片和扫描网格 | 真实照片 | 覆盖漫反射、亮面和镜面金属。扫描时对物体喷过显像剂，照片本身是未喷涂的实拍 |
| NeRO GlossyReal | 十余个真实反光物体 | 真实多视角照片 | 用来展示 COLMAP 在反光物体上失败的常用测试 |

TexVerse 来自 Sketchfab，已经限定可再分发的 CC 许可、贴图至少 1024，并排除了带 NoAI 标记的模型。它单独给出 PBR 模型的 id 列表和文字描述，所以可以先筛再下：描述里保留 metal、chrome、steel、ceramic、porcelain、glass，再用金属度、粗糙度贴图确认表面确实亮，底色贴图平坦的算弱纹理。两条都满足的才进入渲染。全量下载 15.8 万个模型没有必要。来源：<https://github.com/yiboz2001/TexVerse>，数据在 Hugging Face `YiboZhang2001/TexVerse`。

ABO 渲成镀铬只能说明「假如这个家具是镀铬的」。队列的品类统计来自 `manifests/abo_queue.json`。glTF 在带有金属度贴图时会把 `metallicFactor` 默认成 1，不能拿这个字段统计有多少金属物体。

评测用真实拍摄，不用 TexVerse 自己留出的模型代替。Stanford-ORB（<https://stanfordorb.github.io/>）、NeRO 的 GlossyReal（<https://github.com/liuyuan-pal/NeRO>）、DTC 的 DSLR 序列（<https://github.com/facebookresearch/DigitalTwinCatalog>）都是几十个物体的量级，只用于最后验收。

玻璃先不放进第一批。TexVerse 的 PBR 里有透明材质，但透明物体上 COLMAP 会不会失败，本项目还没有测过。第一批筛选只收金属和陶瓷。

这些数据集都还没下载，筛选也还没做。

## 数据位置

| 路径 | 内容 | 大小 | 状态 |
| --- | --- | --- | --- |
| `D:\TextureAnything\datasets\abo\abo-3dmodels.tar` | ABO 原始三维模型压缩包 | 154 GB | 原始数据 |
| `D:\TextureAnything\datasets\abo\abo-listings.tar`、`listings` | ABO 商品元数据 | 0.1 GB | 原始数据 |
| `D:\TextureAnything\datasets\abo\models` | 从压缩包解出的 1673 个 glb | 20.6 GB | 在用 |
| `D:\TextureAnything\datasets\abo\p0_queue.json` | 1673 个物体的队列，带 Windows 路径 | 小 | 原件，脚本已改读 `manifests/abo_queue.json` |
| `D:\TextureAnything\datasets\abo\albedo_scores.jsonl` | 7953 个模型的贴图打分 | 小 | 筛选记录 |
| `D:\TextureAnything\datasets\hdri` | 20 张 2k HDRI 和清单 | 0.12 GB | 在用 |
| `D:\TextureAnything\datasets\abo\renders` | 灰色摄影棚 P0/P4，24 帧，512×512 | 119 GB | 不再使用 |
| `D:\TextureAnything\datasets\abo\renders_black` | 中途停掉的黑背景渲染，102 个物体目录 | 7.2 GB | 作废 |
| `D:\TextureAnything\datasets\abo\renders_test` | 3 个物体的 EEVEE 黑背景测试 | 0.09 GB | 参考 |
| `D:\TextureAnything\datasets\abo\renders_reflect` | 反光材质与散斑实验 | 1.7 GB | 最新实验 |
| `D:\TextureAnything\datasets\abo\probe`、`probe_render`、`probe_reflect` | 调试输出 | 小 | 可删除 |
| `D:\TextureAnything\indoors` | Infinigen 室内场景与配对 | 1.9 GB | 不再使用 |
| `E:\TextureAnything-data.zip` | 打包的数据集（不含渲染和调试目录），另含 `data\gso_assets` | 175 GB | 迁移用 |
| 项目根目录下的 `data\` | GSO 资产和早期实验的渲染 | 约 26 GB | 早期实验 |
| 项目根目录下的 `results\` | 各阶段的指标表和报告 | 小 | 早期与最新实验 |
| `dist\gpu_render_task`、`dist\gpu_render_task.zip` | 给服务器的渲染任务包，脚本停在改用 EEVEE 之前的版本 | 小 | 已过时 |

工具位置：Blender `D:\blender\blender-5.2.2-windows-x64`，COLMAP `D:\colmap\bin\colmap.exe`，路径写在 `configs\render.yaml` 和 `configs\colmap.yaml`。
