# YOLO11n 两小时 CPU 召回率优先训练报告

## 结论

- 等级：`effective_improvement`
- 完成 epoch：12
- 训练耗时：102.01 分钟
- 选中检查点：`epoch8.pt`
- 推荐研究/演示阈值：`0.20`

所有检查点与三个候选阈值均未达到 Precision ≥0.70；因此按预定兜底规则选择 F1 最高组合，不能视为达到部署精度要求。

## 标准验证指标

| 指标 | 上轮 | 本轮 | 变化 |
| --- | ---: | ---: | ---: |
| Precision | 0.6534 | 0.5218 | -0.1317 |
| Recall | 0.3800 | 0.6162 | +0.2362 |
| F1 | 0.4806 | 0.5651 | +0.0845 |
| mAP@0.5 | 0.4977 | 0.5679 | +0.0703 |
| mAP@0.5:0.95 | 0.3086 | 0.3756 | +0.0670 |

## IoU=0.50 固定阈值统计

| conf | TP | FP | FN | Precision | Recall | F1 | FP/image |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.15 | 6568 | 7475 | 2591 | 0.4677 | 0.7171 | 0.5662 | 35.0939 |
| 0.20 | 5948 | 5363 | 3211 | 0.5259 | 0.6494 | 0.5811 | 25.1784 |
| 0.25 | 5286 | 3863 | 3873 | 0.5778 | 0.5771 | 0.5775 | 18.1362 |

阈值选择规则为：先要求 Precision ≥ 0.70，再优先选择 Recall；Recall 相差不足0.01时比较F1和mAP@0.5。

本轮以提升 Recall 为首要目标，`conf=0.20` 在候选范围内取得最高 F1。`conf=0.15` 虽然 Recall 更高，但误检明显增加；`conf=0.25` 精度更高，但召回下降。当前模型仍应作为研究验证模型，不宜直接用于无人机自主清洗决策。

## 人工抽查

- 常规窗户和大面积玻璃幕墙的检出数量明显增加。
- 密集幕墙中的框和文字严重重叠，说明模型能发现更多小窗格，但误检与重复视觉干扰仍较多。
- 部分建筑渲染图和弱纹理墙面仍没有检测框。
- 相比一小时模型，“没有框”的问题已经明显减少，当前主要矛盾转为高召回伴随较多误检。

## 交付物

- `runs/yolo11n_window_cpu_2h_recall_20260929/weights/best_recall.pt`
- `runs/yolo11n_window_cpu_2h_recall_20260929/weights/best.pt`
- `runs/yolo11n_window_cpu_2h_recall_20260929/weights/last.pt`
- `runs/yolo11n_window_cpu_2h_recall_20260929_evaluation/BoxPR_curve.png`
- `runs/yolo11n_window_cpu_2h_recall_20260929_evaluation/BoxF1_curve.png`
- `runs/yolo11n_window_cpu_2h_recall_20260929_evaluation/confusion_matrix_iou50_selected.png`
- `runs/yolo11n_window_cpu_2h_recall_20260929_sample20/sample20_contact_sheet.jpg`
- `reports/two_hour_cpu_metrics.json`
- `reports/two_hour_cpu_runtime.json`
