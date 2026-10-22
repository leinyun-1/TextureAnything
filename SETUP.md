# 新机器上的环境与数据

在一台新机器上，从克隆仓库到能跑实验分四步：装环境、设路径、下数据、自检。数据不在 Git 里，所有脚本通过三个环境变量找到数据和工具。

## 1. 克隆与 Python 依赖

```bash
git clone https://github.com/leinyun-1/TextureAnything.git
cd TextureAnything
python -m pip install -r requirements.txt
```

建议用 Python 3.10 或 3.11。没有显示器的 Linux 服务器上，把 `opencv-python` 换成 `opencv-python-headless`。

## 2. Blender 和 COLMAP

两者都用固定版本，渲染和重建结果才能和本机对得上。

| 工具 | 版本 | 下载 |
| --- | --- | --- |
| Blender | 5.2.2 | Linux：`https://download.blender.org/release/Blender5.2/blender-5.2.2-linux-x64.tar.xz`<br>Windows：`https://download.blender.org/release/Blender5.2/blender-5.2.2-windows-x64.zip` |
| COLMAP | 4.2.0 | Windows：GitHub 发布页的 `colmap-x64-windows-cuda.zip`（`https://github.com/colmap/colmap/releases/tag/4.2.0`）<br>Linux：`conda install -c conda-forge colmap=4.2.0` |

Blender 解压即可使用，不需要安装。COLMAP 的 GPU 特征提取失败时，脚本会自动改用 CPU。

Linux 示例：

```bash
wget https://download.blender.org/release/Blender5.2/blender-5.2.2-linux-x64.tar.xz
tar -xf blender-5.2.2-linux-x64.tar.xz -C /opt
conda install -y -c conda-forge colmap=4.2.0
```

## 3. 路径

| 变量 | 含义 | 不设时的默认值 |
| --- | --- | --- |
| `TA_DATA_ROOT` | 数据根目录，下面有 `abo/`、`hdri/`，渲染输出也写在这里 | `D:\TextureAnything\datasets` |
| `TA_BLENDER` | Blender 可执行文件 | `configs/render.yaml` 里的 `blender_exe` |
| `TA_COLMAP` | COLMAP 可执行文件 | `configs/colmap.yaml` 里的 `colmap_exe` |

默认值是制作机器上的路径，其他机器都要设置这三个变量。数据根目录至少预留 25 GB，渲染输出另算。

Linux：

```bash
export TA_DATA_ROOT=/data/TextureAnything
export TA_BLENDER=/opt/blender-5.2.2-linux-x64/blender
export TA_COLMAP=$(which colmap)
```

Windows PowerShell（`setx` 写入后对新开的终端生效）：

```powershell
setx TA_DATA_ROOT "E:\TextureAnything\datasets"
setx TA_BLENDER "E:\blender\blender-5.2.2-windows-x64\blender.exe"
setx TA_COLMAP "E:\colmap\bin\colmap.exe"
```

## 4. 下载数据

```bash
python scripts/fetch_data.py
```

这条命令会下载两部分：

- **ABO 模型**：`manifests/abo_queue.json` 里列出的 1673 个物体，每个一个 glb，按单个文件从 ABO 的公开 S3 下载，共约 20.6 GB，许可 CC-BY-4.0。用不着下载 154 GB 的整包。
- **HDRI**：20 张 Poly Haven 室内 HDRI，2k 分辨率，约 121 MB，许可 CC0。

下载完的目录结构：

```text
$TA_DATA_ROOT/
├── abo/models/<物体id>.glb
└── hdri/<名字>_2k.hdr、manifest.json
```

下载中断后重跑同一条命令即可，已经下好的文件会跳过。半截文件先写成 `.part`，完整后才改名。常用参数：

```bash
python scripts/fetch_data.py hdri                      # 只下 HDRI
python scripts/fetch_data.py abo --limit 20            # 只下队列前 20 个物体
python scripts/fetch_data.py abo --ids B07BG69MRW B07BLB6644
python scripts/fetch_data.py abo --workers 16          # 更多并发
```

如果手边有制作机器打包的 `TextureAnything-data.zip`，解压到 `TA_DATA_ROOT` 也可以，内容是上面两部分的超集。

## 5. 自检

```bash
python scripts/check_setup.py
```

它会检查 Python 包、Blender 和 COLMAP 能否运行、ABO 模型和 HDRI 是否齐全。缺什么就在最后列出补救命令。全部通过时输出 `All set.`。

再用一个物体跑一遍端到端，确认渲染和重建都正常，几分钟就能跑完：

```bash
python scripts/exp_reflective_colmap.py --ids B07BG69MRW --materials chrome --results-name smoke.csv
```

渲染输出在 `$TA_DATA_ROOT/abo/renders_reflect/B07BG69MRW/chrome`，结果表在 `results/reflect_colmap/smoke.csv`。这是镀铬双层床，正常情况下几乎所有帧都能注册，但旋转误差很大（制作机器上是 80/81 帧、86.75°），也就是反光物体“看起来成功、位姿却错”的典型情况。没有显示器的 Linux 机器上，EEVEE 需要 GPU 驱动支持无头渲染。如果这一步渲不出图，先查驱动。

## 不在默认下载里的数据

下面这些只在复现早期实验、或重新筛选物体时才需要，各阶段的情况见 `HISTORY.md`。

- **ABO 完整压缩包和商品元数据**：用于重新生成物体队列。下载 `https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-3dmodels.tar`（154 GB）和 `https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar`，放到 `$TA_DATA_ROOT/abo/`，并把 listings 解压到 `abo/listings/`。然后依次运行 `scripts/screen_abo.py`（打分）和 `scripts/prepare_abo.py`（解出模型、重写 `manifests/abo_queue.json`）。
- **Google Scanned Objects**：早期 GSO 实验用的。运行 `python scripts/download_gso.py`，下载到项目内的 `data/gso_assets`。
- **室内场景**：Infinigen 生成，已经不再使用，没有下载脚本。
