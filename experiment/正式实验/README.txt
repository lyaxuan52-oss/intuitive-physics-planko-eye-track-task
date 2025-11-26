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

### 2. 生成刺激（离线）
```bash
python stimulus_generator.py
```
这会生成 `stimuli/stimulus_config.csv`，包含所有试次的配置、SU/HV值、真实轨迹等。

**注意**: 生成时会自动过滤：
- 球没有落入任何接球器的试次（outcome='none'）
- 主启发式为'ambiguous'的试次
- 最终保证每个HV×SU组合恰好40个有效试次

### 3. 运行实验（在线）
```bash
python experiment_runtime.py
```
实验开始前会弹窗收集被试信息（ID、性别、年龄、球类经验）。

## 实验设计

### 刺激参数
- **球的起始位置**: 固定在屏幕水平中心（960px）
- **接球器布局**: 
  - 左接球器: 472-872px（宽400px）
  - 右接球器: 1048-1448px（宽400px）
  - 中间间隙: 176px（球必须被挡板偏转才能落入接球器）
- **挡板**: 10个随机分布的挡板（100px × 10px）

### 启发式定义
- **主启发式（直线投影）**: 
  - 从球的起始位置到左/右接球器中心是否有直线路径
  - 只有一侧有直线 → 预测该侧 → congruent/conflict
  - 两侧都有/都没有 → ambiguous（生成时过滤）
- **次启发式1（挡板数量）**: 哪侧挡板数量多，预测落哪侧
- **次启发式2（第一次碰撞）**: 第一个打到的挡板在哪侧，预测落哪侧

### 试验条件
- **条件**: 2 HV (congruent/conflict) × 5 SU levels = 10 combinations
- **试次数**: 每种组合40个试次，共400试次
- **试次分配**: 随机打乱后平均分配到5个blocks，每个block约80个试次
- **试次顺序**: 每次运行实验均完全随机
- **Block间休息**: 1-2分钟，重新眼动校准

## 实验流程

### 单个试次
1. **注视点**: 800±200ms
2. **呈现挡板**: 显示挡板配置，被试预测球的落点
3. **选择**: 
   - 按 **F** 键: 预测落入左侧（绿色）
   - 按 **J** 键: 预测落入右侧（蓝色）
4. **信心度报告**: 
   - 用数字键输入0-100的值（例如输入 7 5 → 75%）
   - 按退格键删除，按回车键确认
5. **反馈**: 播放球的真实下落动画
6. **试次间隔**: 1000ms

## 数据输出

### `stimuli/stimulus_config.csv`
刺激配置文件，包含：
- BoardID, SU_raw, SU_level
- HV_primary, HV_secondary_count, HV_secondary_collision
- PhysOutcome（真实物理落点）
- Ball_start_x, Ball_start_y
- True_trajectory（球的真实轨迹，JSON格式）
- Planks（挡板配置，JSON格式）

### `data/sub-{ID}/behavioral.csv`
行为数据，包含：
- **被试信息**: SubjectID, Gender, Age, BallExperience
- **试次信息**: TrialID, BlockID, BoardID
- **刺激参数**: SU_level, HV_primary, HV_secondary_count, HV_secondary_collision, PhysOutcome
- **行为数据**: Choice, RT, Correct, Confidence, Confidence_RT
- **时间戳**: Stim_onset, Response_time, Feedback_onset, Ball_drop_start, Ball_drop_end

### `data/sub-{ID}/{ID}.edf`
Eyelink眼动数据（需SR Research软件解析）
