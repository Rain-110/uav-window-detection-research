# 玻璃幕墙数据集清洗与准备报告

## 执行结论

- 原始数据保持不变，所有处理结果位于独立项目目录。
- 六类清洗集：训练 3220 张，验证 213 张。
- 单类清洗集：训练 3220 张，验证 213 张；标签中仅保留 Glass Window（ID 0）。
- 自动删除零面积标注行 39 条。
- 自动隔离同集合完全重复样本 215 张。
- 自动隔离跨训练/验证重复的训练样本 2 张。
- 已生成标注预览 30 张。

## 需要人工确认

- duplicate_label_conflict: 32 张图片，已生成联系表并复制到隔离目录。
- empty_label_review: 53 张图片，已生成联系表并复制到隔离目录。
- missing_label: 14 张图片，已生成联系表并复制到隔离目录。

## 隔离文件分类

- cross_split_duplicate: 4 个文件副本。
- duplicate_label_conflict: 64 个文件副本。
- empty_label_review: 106 个文件副本。
- exact_duplicate: 430 个文件副本。
- missing_label: 14 个文件副本。
- orphan_label: 5 个文件副本。

## 生成文件

- `reports/file_manifest.csv`：原始文件清单和 SHA256。
- `reports/audit_report.json`：机器可读审计结果。
- `reports/issues_and_fixes.csv`：问题及处理动作。
- `reports/quarantine_manifest.csv`：隔离文件索引。
- `reports/preview_index.csv`：标注预览索引。
- `configs/data_clean_6class.yaml`：六类训练配置。
- `configs/data_clean_window_only.yaml`：Glass Window 单类训练配置。

## 重要说明

- 隔离目录保存的是原文件副本，原始数据没有删除或覆盖。
- 单类数据集图片通过 NTFS 硬链接复用六类清洗集图片；标签是独立文件。
- 在人工确认空标签和冲突重复样本前，不把这些文件加入训练集。
