# 高空玻璃幕墙清洗无人机：视觉识别验证

本仓库记录第一次任务（截至 2026-09-29）的玻璃窗检测研究。目标是验证 YOLO11n 在普通 CPU 上识别幕墙窗格的可行性；目前属于研究验证，尚不能用于无人机自主清洗决策。

## 已完成

- 审计公开玻璃幕墙数据：清洗后保留训练集 3,220 张、验证集 213 张；单类标签分别含 80,467 和 9,159 个窗格。隔离重复、缺失或有冲突的样本，并保留审计记录。
- 完成 YOLO11n 的 CPU 冒烟测试、1 小时训练及以召回率为目标的约 2 小时续训。
- 续训模型在标准验证中的 Precision 为 **0.5218**、Recall 为 **0.6162**、mAP@0.5 为 **0.5679**。相比上一轮召回率提高 0.2362，但精度下降，仍未达到部署要求。

详细过程与局限见 [`first_task/results/`](first_task/results/)。

## 内容与数据

| 路径 | 内容 |
| --- | --- |
| `first_task/scripts/` | 数据审计、校验、训练与评估脚本 |
| `first_task/results/` | 第一次任务的审计和实验结果 |
| `dataset/` | 数据来源、类别和获取说明 |

原始图像来自 [monymon/dataset_glasswindow](https://github.com/monymon/dataset_glasswindow)。上游仓库未声明再分发许可证，因此本仓库只提供数据说明与处理脚本，不镜像图像、模型权重或大型训练输出。完整数据请从上游获取，并遵守其使用条件。仿真代码另见 [原作者仓库](https://github.com/g-kohl/UAV-Window-Detection)。

## 复现

1. 安装 Python 3.10+，再安装 `pillow`、`ultralytics`、`torch`、`psutil`。
2. 下载上游数据，将其 `train/`、`test/` 等文件放在 `dataset/raw/`。
3. 从 Ultralytics 下载 `yolo11n.pt` 并放入 `first_task/`。在仓库根目录运行 `python first_task/scripts/audit_and_prepare.py`。脚本会生成清洗数据及配置；如输出目录已有内容，脚本会拒绝覆盖。
4. 运行 `python first_task/scripts/verify_prepared.py` 校验数据。训练脚本会使用该 YOLO11n 预训练权重；CPU 训练耗时较长，具体配置和结果见报告。

实验指标来自本地历史记录，本仓库没有上传模型权重，无法直接复现已训练模型的推理结果。
