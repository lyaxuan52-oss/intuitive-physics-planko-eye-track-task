"""
刺激生成脚本
运行此脚本生成所有试次的刺激配置和SU/HV值
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
try:
    import pymunk
except Exception:
    pymunk = None
    print("Warning: 'pymunk' is not installed or could not be imported; install it with 'pip install pymunk' to run physics simulations.")
from multiprocessing import Pool, cpu_count

from config import *
from physics_utils import (
    create_space, create_ball, create_plank, create_catchers,
    run_simulation, generate_random_planks,
    check_straight_line_heuristic, count_planks_per_side
)


def apply_jitter_to_planks(planks):
    """对挡板配置应用jitter扰动"""
    jittered = []
    for p in planks:
        jittered.append({
            'x': p['x'] + np.random.normal(0, JITTER_POSITION_STD),
            'y': p['y'] + np.random.normal(0, JITTER_POSITION_STD),
            'angle': p['angle'] + np.random.normal(0, JITTER_ANGLE_STD)
        })
    return jittered


def calculate_su(planks, start_side):
    """计算主观不确定性（SU）

    SU 的定义：在固定起点（左侧或右侧）下，对同一挡板配置反复施加 jitter，
    观察球最终落入左右接球器的比例。
    """
    left_count = 0
    right_count = 0

    # 根据起点侧确定球的x坐标
    if start_side == 'left':
        start_x = CATCHER_LEFT_X
    else:
        start_x = CATCHER_RIGHT_X

    for _ in range(N_JITTER_SIMULATIONS):
        space = create_space()
        ball_body, ball_shape = create_ball(space, start_x, BALL_START_Y)
        create_catchers(space)

        # 应用jitter
        jittered_planks = apply_jitter_to_planks(planks)
        for plank in jittered_planks:
            create_plank(space, plank['x'], plank['y'], plank['angle'])

        # 新的run_simulation需要ball_shape，并依靠接球器传感器判定落点
        outcome, _, _ = run_simulation(space, ball_body, ball_shape, jittered_planks)
        if outcome == 'left':
            left_count += 1
        elif outcome == 'right':
            right_count += 1
    
    # SU = 结果分布的熟（entropy-like）
    total = left_count + right_count
    if total == 0:
        return 0.5
    
    p_left = left_count / total
    p_right = right_count / total
    
    # 使用简化的熟公式：min(p_left, p_right) * 2
    # 这样SU范围是0-1，0.5表示最不确定
    su = min(p_left, p_right) * 2
    
    return su


def determine_hv(planks, phys_outcome, first_collision_side, start_side):
    """判断启发式有效性

    返回: (hv_primary, hv_secondary_count, hv_secondary_collision)
        hv_primary: 直线落下启发式 ('congruent', 'conflict', or 'ambiguous')
        hv_secondary_count: 挡板数量启发式 ('congruent', 'conflict', or 'equal')
        hv_secondary_collision: 第一次碰撞启发式 ('congruent', 'conflict', or 'none')

    主启发式（按你现在的定义）：
    - 球真实地从左侧或右侧接球器正上方下落（由 start_side 指定）。
    - 启发式想象：沿着这条"竖直路径"直接落到同一侧接球器，如果路径上**没有任何挡板**，
      那么启发式会预测球会落在起点这一侧；
    - 如果这条竖直路径上被挡板截断，则主启发式对该板是"ambiguous"，

    hv_primary 判定：
    - 若竖直路径被挡（直线上有挡板） → hv_primary = 'ambiguous'
    - 若竖直路径无遮挡：
        · 如果物理结果 phys_outcome == start_side → 'congruent'
        · 如果物理结果落到另一侧 → 'conflict'（看上去能直线落入，对微小扰动敏感，实际被弹到另一侧）
    """

    # 1. 检查当前起点侧的"竖直直线路径"上是否有挡板
    if start_side == 'left':
        straight_clear = check_straight_line_heuristic(
            CATCHER_LEFT_X, BALL_START_Y,
            CATCHER_LEFT_X, CATCHER_Y,
            planks
        )
    else:
        straight_clear = check_straight_line_heuristic(
            CATCHER_RIGHT_X, BALL_START_Y,
            CATCHER_RIGHT_X, CATCHER_Y,
            planks
        )

    # 如果竖直路径被挡，主启发式记为 ambiguous
    if not straight_clear:
        hv_primary = 'ambiguous'
    else:
        # 竖直路径无遮挡 → 启发式预测 = start_side
        if phys_outcome not in ('left', 'right'):
            # 理论上已经在外部过滤，这里保险起见
            hv_primary = 'ambiguous'
        elif phys_outcome == start_side:
            hv_primary = 'congruent'
        else:
            hv_primary = 'conflict'

    # 次启发式1：挡板数量
    left_count, right_count = count_planks_per_side(planks)
    
    if left_count > right_count:
        heur_pred_count = 'left'  # 左边挡板多，预测落左
    elif right_count > left_count:
        heur_pred_count = 'right'  # 右边挡板多，预测落右
    else:
        heur_pred_count = 'equal'  # 挡板数量相等

    if heur_pred_count == 'equal':
        hv_secondary_count = 'equal'
    elif heur_pred_count == phys_outcome:
        hv_secondary_count = 'congruent'
    else:
        hv_secondary_count = 'conflict'

    # 次启发式2：第一次碰撞
    if first_collision_side == 'none':
        hv_secondary_collision = 'none'
    elif first_collision_side == phys_outcome:
        hv_secondary_collision = 'congruent'
    else:
        hv_secondary_collision = 'conflict'

    return hv_primary, hv_secondary_count, hv_secondary_collision


def generate_board_for_condition(target_hv, target_su_min, target_su_max, max_attempts=1000):
    """
    为特定HV×SU组合生成一个板配置
    target_hv: 'congruent', 'conflict' 或 'ambiguous'
    target_su_min, target_su_max: SU目标范围
    """
    for attempt in range(max_attempts):
        # 随机生成挡板配置
        planks = generate_random_planks(N_PLANKS, seed=None)

        # 随机决定本块板的起点在左侧还是右侧
        start_side = np.random.choice(['left', 'right'])
        start_x = CATCHER_LEFT_X if start_side == 'left' else CATCHER_RIGHT_X

        # 计算SU：基于这一侧的起点
        su_raw = calculate_su(planks, start_side)
        
        # 检查SU是否在目标范围内
        if not (target_su_min <= su_raw < target_su_max):
            continue
        
        # 运行一次模拟获取物理结果和轨迹
        space = create_space()
        ball_body, ball_shape = create_ball(space, start_x, BALL_START_Y)
        create_catchers(space)
        for plank in planks:
            create_plank(space, plank['x'], plank['y'], plank['angle'])
        
        # 使用基于接球器碰撞的严格落点判定
        outcome, trajectory, first_collision_side = run_simulation(space, ball_body, ball_shape, planks)
        
        # 过滤掉'none'结果
        if outcome == 'none':
            continue
        
        # 判断HV
        hv_primary, hv_secondary_count, hv_secondary_collision = determine_hv(
            planks, outcome, first_collision_side, start_side
        )

        # 检查主启发式是否符合目标
        if hv_primary == target_hv:
            return {
                'planks': planks,
                'su_raw': su_raw,
                'hv_primary': hv_primary,
                'hv_secondary_count': hv_secondary_count,
                'hv_secondary_collision': hv_secondary_collision,
                'phys_outcome': outcome,
                'trajectory': trajectory,
                'ball_start_side': start_side,
                'ball_start_x': start_x
            }
    
    # 如果达到最大尝试次数仍未成功，返回None
    return None


def _generate_single_board(args):
    """并行计算的包装函数"""
    hv_type, su_min, su_max, board_id, su_level = args
    board = generate_board_for_condition(hv_type, su_min, su_max)
    if board is not None:
        board['BoardID'] = f"B{board_id:04d}"
        board['SU_level'] = su_level
    return board


def generate_all_stimuli(use_parallel=True):
    """
    生成所有刺激配置
    
    分布策略：
    - ambiguous 和 congruent: 每个 SU level 严格生成 TRIALS_PER_CONDITION 个试次
    - conflict: 总共生成约 TRIALS_PER_CONDITION * SU_LEVELS 个试次，SU 分布不严格要求均匀
    """
    print("=" * 50)
    print("开始生成刺激配置...")
    print("=" * 50)
    
    # 定义SU等级边界
    su_bins = np.linspace(0, 1, SU_LEVELS + 1)
    
    # 准备所有任务参数
    tasks = []
    board_id = 0
    
    # 对 ambiguous 和 congruent：严格按 SU level 生成
    for hv_type in ['congruent', 'ambiguous']:
        for su_level in range(1, SU_LEVELS + 1):
            su_min = su_bins[su_level - 1]
            su_max = su_bins[su_level]
            for _ in range(TRIALS_PER_CONDITION):
                tasks.append((hv_type, su_min, su_max, board_id, su_level))
                board_id += 1
    
    # 对 conflict：生成总数接近的试次，但不严格限制每个 SU level
    # 总数 = TRIALS_PER_CONDITION * SU_LEVELS（与 ambiguous/congruent 总数一致）
    n_conflict_trials = TRIALS_PER_CONDITION * SU_LEVELS
    print(f"\nambiguous/congruent: 每个 SU level {TRIALS_PER_CONDITION} 个试次，共 {TRIALS_PER_CONDITION * SU_LEVELS} 个")
    print(f"conflict: 总共约 {n_conflict_trials} 个试次，SU 分布不严格限制")
    
    for _ in range(n_conflict_trials):
        # 随机选择一个 SU level（允许不均匀分布）
        su_level = np.random.randint(1, SU_LEVELS + 1)
        su_min = su_bins[su_level - 1]
        su_max = su_bins[su_level]
        tasks.append(('conflict', su_min, su_max, board_id, su_level))
        board_id += 1
    
    print(f"\n总共需要生成 {len(tasks)} 个刺激配置")
    
    # 并行生成
    if use_parallel:
        n_cores = cpu_count()
        print(f"使用 {n_cores} 个CPU核心并行生成...")
        
        with Pool(processes=n_cores) as pool:
            all_boards = list(tqdm(
                pool.imap(_generate_single_board, tasks),
                total=len(tasks),
                desc="生成刺激"
            ))
    else:
        # 串行生成（调试用）
        all_boards = []
        for task in tqdm(tasks, desc="生成刺激"):
            board = _generate_single_board(task)
            all_boards.append(board)
    
    # 过滤掉None（生成失败的）
    all_boards = [b for b in all_boards if b is not None]
    return all_boards


def save_stimuli(boards, output_dir='stimuli'):
    """保存刺激配置到CSV文件"""
    # 使用脚本所在目录的绝对路径
    script_dir = Path(__file__).parent
    output_path = script_dir / output_dir
    output_path.mkdir(exist_ok=True)
    
    # 准备DataFrame
    records = []
    for board in boards:
        record = {
            'BoardID': board['BoardID'],
            'SU_raw': board['su_raw'],
            'SU_level': board['SU_level'],
            'HV_primary': board['hv_primary'],
            'HV_secondary_count': board['hv_secondary_count'],
            'HV_secondary_collision': board['hv_secondary_collision'],
            'PhysOutcome': board['phys_outcome'],
            'Ball_start_x': board['ball_start_x'],  # 每块板自己的真实起点
            'Ball_start_y': BALL_START_Y,
            'Ball_start_side': board['ball_start_side'],
            # 轨迹和挡板配置
            'True_trajectory': json.dumps(board['trajectory']),
            'Planks': json.dumps(board['planks'])  # 可选，用于调试或重现
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # 保存到CSV
    output_file = output_path / 'stimulus_config.csv'
    df.to_csv(output_file, index=False)
    print(f"\n刺激配置已保存到: {output_file}")
    
    # 打印统计信息
    print("\n" + "=" * 50)
    print("刺激配置统计:")
    print("=" * 50)
    print("\nHV_primary分布:")
    print(df['HV_primary'].value_counts())
    print("\nHV_secondary_count分布:")
    print(df['HV_secondary_count'].value_counts())
    print("\nHV_secondary_collision分布:")
    print(df['HV_secondary_collision'].value_counts())
    print("\nSU_level分布:")
    print(df['SU_level'].value_counts().sort_index())
    print("\nPhysOutcome分布:")
    print(df['PhysOutcome'].value_counts())
    print("\nHV_primary × SU_level组合分布:")
    print(df.groupby(['HV_primary', 'SU_level']).size().unstack(fill_value=0))
    
    return output_file


if __name__ == '__main__':
    # 设置随机种子以便复现（可选）
    # np.random.seed(42)
    
    # 生成刺激
    boards = generate_all_stimuli()
    
    # 保存
    save_stimuli(boards)
    
    print("\n✓ 刺激生成完成！")
    print("下一步: 运行 experiment_runtime.py 开始实验")
