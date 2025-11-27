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

### 2. 生成正式实验刺激（离线）
```bash
python stimulus_generator.py
```
这会生成 `stimuli/stimulus_config.csv`，包含所有试次的配置和SU/HV值。

### 3. 三阶段实验脚本概览

1. **训练练习（无眼动、无数据记录，仅演示任务）**  
   脚本：`训练/practice_demo.py`  
   - 呈现若干示例板子和真实轨迹；  
   - 被试通过 **鼠标点击左/右接球器** 做出预测；  
   - 不记录行为数据。

2. **启发式行为实验（挡板数量启发式，仅行为）**  
   脚本：`启发式/run_count_heuristic_experiment.py`  
   - 呈现若干板子；  
   - 被试通过 **鼠标点击左/右接球器** 做出二选一判断，不报告信心度；  
   - 只保存行为数据到 `启发式/data_count_heuristic/`。

3. **正式眼动实验**  
   脚本：`正式实验/experiment_runtime.py`  
   - 启动前连接Eyelink主机并完成校准；  
   - 每个trial中，被试先用 **鼠标点击左/右接球器** 预测落点，随后通过 **鼠标滑条 +“确认”按钮** 报告 0–100% 信心度；  
   - 同时记录行为数据和EDF眼动文件。

> 说明：在三个阶段中，被试的选择一律使用 **鼠标点击** 完成；键盘主要用于开始/结束实验（如空格、ESC），不再使用F/J或数字键进行选择与信心度输入。

## 实验设计
- **条件**: 2 HV (congruent/conflict) × 5 SU levels = 10 combinations
- **试次数**: 每种组合20个试次（可配置），共200试次
- **分组**: 每2个组合为1个block，共5个blocks

## 数据输出
- `stimuli/stimulus_config.csv`: 刺激配置
- `data/sub-{ID}/behavioral.csv`: 行为和时间戳数据
- `data/sub-{ID}/{ID}.edf`: Eyelink眼动数据


【ps】补充！！：因为模拟试次生成有最大尝试次数，所以有时候有些条件和组合的试次会缺少，补充了检查文件.py和安全补充试次.py，
