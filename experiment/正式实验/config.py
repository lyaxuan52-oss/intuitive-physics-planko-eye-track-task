"""
实验配置参数
"""

# ============ 物理引擎参数 ============
# 设计分辨率（用于刺激生成时的基准）
DESIGN_WIDTH = 1920
DESIGN_HEIGHT = 1080

# 运行时分辨率（会在实验开始时根据实际屏幕尺寸更新）
SCREEN_WIDTH = DESIGN_WIDTH
SCREEN_HEIGHT = DESIGN_HEIGHT
FPS = 60

# 球参数
BALL_RADIUS = 10  # pixels
BALL_START_Y = 100
BALL_MASS = 1.0

# 水平起点的默认值（当前真实刺激生成时，球在每个trial会从左/右接球器正上方的固定x位置下落；
# 这里的BALL_START_X主要作为默认/回退值使用）
BALL_START_X = SCREEN_WIDTH // 2

# 挡板参数
PLANK_WIDTH = 100  # pixels
PLANK_HEIGHT = 10
PLANK_MASS = 10.0

# 接球器参数（宽大的接球器，间隙比之前更小，便于产生“看起来能进但实际被弹走”的情况）
CATCHER_WIDTH = 360  # 稍微缩窄以便把接球器靠拢
CATCHER_HEIGHT = 30
CATCHER_Y = SCREEN_HEIGHT - 100  # 980
# 左右接球器位置：相比之前更靠中间，但仍保持左右分离
CATCHER_LEFT_X = SCREEN_WIDTH * 0.40  # 768
CATCHER_RIGHT_X = SCREEN_WIDTH * 0.60  # 1152
# 接球器布局（宽度360px）:
# 左接球器: 588-948px, 右接球器: 972-1332px
# 中心间隙: 948-972 = 24px（间隙明显缩小，更容易出现“直线看似落入一侧但被反弹到另一侧”的情形）
# 布局特点：接球器大且更靠拢，挡板可以把球从一侧弹到另一侧；
# 生成时自动过滤没有打到接球器的试次（outcome=='none'）

# 物理参数
GRAVITY = 980  # pixels/s^2 (加快生成速度)
ELASTICITY = 0.9
FRICTION = 0.3

# ============ SU计算参数 ============
N_JITTER_SIMULATIONS = 100  # 100次jitter模拟
JITTER_POSITION_STD = 2  # pixels
JITTER_ANGLE_STD = 1     # degrees
SU_LEVELS = 5  # 将SU值分成5个等级

# ============ 实验设计参数 ============
TRIALS_PER_CONDITION = 30  # 每种HV×SU组合的试次数
HV_TYPES = ['congruent', 'conflict', 'ambiguous']  # 主启发式类型
N_CONDITIONS = len(HV_TYPES) * SU_LEVELS  # 2 × 5 = 10
COMBINATIONS_PER_BLOCK = 10  # 每个block包含所有10种组合（充分混合）
N_BLOCKS = 5  # 5个blocks，每个block包含每种组合的8个试次（40÷5=8）

# ============ 时间参数（ms）============
FIXATION_DURATION = 800
FIXATION_JITTER = 200  # ±200ms
ITI_DURATION = 1000
# ============ 刺激生成参数 ============
N_PLANKS=10
PLANK_Y_MIN = 200
PLANK_Y_MAX = SCREEN_HEIGHT - 200


# ============ 屏幕自适应函数 ============
def update_screen_params(actual_width, actual_height):
    """
    根据实际屏幕尺寸更新所有坐标和尺寸参数
    
    Args:
        actual_width: 实际屏幕宽度
        actual_height: 实际屏幕高度
    """
    global SCREEN_WIDTH, SCREEN_HEIGHT
    global BALL_START_X, BALL_START_Y, BALL_RADIUS
    global PLANK_WIDTH, PLANK_HEIGHT
    global CATCHER_WIDTH, CATCHER_HEIGHT, CATCHER_Y
    global CATCHER_LEFT_X, CATCHER_RIGHT_X
    global PLANK_Y_MIN, PLANK_Y_MAX
    global GRAVITY
    
    # 计算缩放比例
    scale_x = actual_width / DESIGN_WIDTH
    scale_y = actual_height / DESIGN_HEIGHT
    scale = min(scale_x, scale_y)  # 使用较小的缩放比例保持宽高比
    
    # 更新屏幕尺寸
    SCREEN_WIDTH = actual_width
    SCREEN_HEIGHT = actual_height
    
    # 更新所有尺寸参数（按比例缩放）
    BALL_RADIUS = int(10 * scale)
    BALL_START_Y = int(100 * scale)
    BALL_START_X = SCREEN_WIDTH // 2
    
    PLANK_WIDTH = int(100 * scale)
    PLANK_HEIGHT = int(10 * scale)
    
    CATCHER_WIDTH = int(360 * scale)
    CATCHER_HEIGHT = int(30 * scale)
    CATCHER_Y = SCREEN_HEIGHT - int(100 * scale)
    CATCHER_LEFT_X = int(SCREEN_WIDTH * 0.40)
    CATCHER_RIGHT_X = int(SCREEN_WIDTH * 0.60)
    
    PLANK_Y_MIN = int(200 * scale)
    PLANK_Y_MAX = SCREEN_HEIGHT - int(200 * scale)
    
    GRAVITY = int(980 * scale)  # 重力也需要缩放以保持相同的物理效果
