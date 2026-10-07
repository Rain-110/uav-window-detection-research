# 数据集说明

来源：[monymon/dataset_glasswindow](https://github.com/monymon/dataset_glasswindow)，包含玻璃窗、立柱、墙体、人员、门和车辆六类标注。原仓库的 `data.yaml` 使用作者机器上的绝对路径，使用前需要重新生成本地配置。

第一次任务从原始数据整理出训练 3,220 张、验证 213 张的六类集，以及相同图像的单类玻璃窗集。处理细节和异常样本数量见 [`../first_task/results/audit_report.md`](../first_task/results/audit_report.md)。

上游未提供明确的再分发许可证，本仓库不包含原始或处理后的图像与标签。请自行从上游获取数据，并确认使用权限。`first_task/scripts/audit_and_prepare.py` 用于在本地重新生成处理结果。
