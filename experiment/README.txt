# Eyelink-1000 Planko Experiment

## 项目结构
```
eyelink_planko/
├── config.py              # 实验配置参数
├── stimulus_generator.py  # 刺激生成脚本（离线运行）
├── experiment_runtime.py  # 实验运行脚本（在线运行）
├── physics_utils.py       # Pymunk物理引擎工具
├── stimuli/               # 生成的刺激文件夹
│   ├── stimulus_config.csv
│   └── boards/            # 保存的板配置（可选）
├── data/                  # 实验数据文件夹
│   └── sub-{ID}/
│       ├── behavioral.csv
│       └── {ID}.edf       # Eyelink数据
└── requirements.txt       # Python依赖
```

## 运行流程

### 1. 安装依赖
```bash
pip install -r requirements.txt
```
我们的眼动部分采用的是python文件+pylink的库，并未使用eyelink1000自带的eb软件进行build

### 2. 生成刺激（离线）
```bash
python stimulus_generator.py
```
这会生成 `stimuli/stimulus_config.csv`，包含所有试次的配置和SU/HV值。

### 3. 运行实验（在线）
```bash
python experiment_runtime.py
```
实验开始前会弹窗收集被试信息（ID、性别、年龄、球类经验）。

## 实验设计
- **条件**: 2 HV (congruent/conflict) × 5 SU levels = 10 combinations
- **试次数**: 每种组合20个试次（可配置），共200试次
- **分组**: 每2个组合为1个block，共5个blocks
- **Block间休息**: 1-2分钟，重新眼动校准

## 数据输出
- `stimuli/stimulus_config.csv`: 刺激配置
- `data/sub-{ID}/behavioral.csv`: 行为和时间戳数据
- `data/sub-{ID}/{ID}.edf`: Eyelink眼动数据


【ps】补充！！：因为模拟试次生成有最大尝试次数，所以有时候有些条件和组合的试次会缺少，补充了检查文件.py和安全补充试次.py，
