# clean_v2数据冻结记录

该目录用于唯一标识基线训练所用的`dataset_2_binary_clean_v2`。

- 文件对：3671
- 数据集总指纹：`e45a10b3ee55117b9769071dfb097e21017f46cee8a35a801c70854c84baa534`
- 清单SHA-256：`1182d65d747bd22afc047f8fbf09cb7036fe0f6cdf18847cb1049d6bbdae8c16`

冻结后不得直接修改clean_v2；发现问题时复制生成clean_v3，并重新分析、复检和冻结。

## 训练前独立验证

在项目根目录执行：

```bat
D:\conda_env\yolov5_7\python.exe tools\verify_frozen_dataset.py ^
  --dataset ..\datasets\dataset_2_binary_clean_v2 ^
  --freeze reports\data_freeze\clean_v2 ^
  --report reports\data_freeze\clean_v2\latest_verification.json
```

返回`PASS`且进程退出码为0才表示数据可以用于正式训练；新增、缺失、内容变化或总指纹不一致都会返回`FAIL`和非0退出码。

## 正式训练入口

不要直接运行`train.py`，正式基线训练统一使用：

```bat
D:\conda_env\yolov5_7\python.exe tools\train_frozen.py [其他训练参数]
```

`train_frozen.py`固定使用`data/3d_printer_binary_v2.yaml`，不允许通过`--data`替换数据配置。它会先执行完整冻结验证；验证失败时不会调用YOLOv5训练程序。
