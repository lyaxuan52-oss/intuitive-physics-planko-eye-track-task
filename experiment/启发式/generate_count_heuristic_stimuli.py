"""\
次启发式（挡板数量启发）专用刺激生成脚本

目标：
- 生成一个单独的刺激文件，用于检验“左右挡板数量差异很大”时的启发式偏好。
- 重点控制：
  - 左右挡板数量差异 |N_left - N_right| 较大；
  - 记下“挡板多的一侧”（CountHeur_side），作为挡板数量启发式的预测侧；
  - 标记该启发式与真实物理结果的关系：HV_count = 'congruent' / 'conflict'。

输出文件：stimuli/stimuli_count_heuristic.csv
  每一行是一块板（一个 trial），包含：
  - BoardID
  - HV_count: 'congruent' 或 'conflict'（基于挡板数量启发式）
  - CountHeur_side: 'left' 或 'right'（挡板较多的一侧）
  - N_planks_left, N_planks_right, PlankCount_diff
  - PhysOutcome: 'left' / 'right'（小球真实落点）
  - Ball_start_side, Ball_start_x, Ball_start_y
  - True_trajectory（JSON 字符串）
  - Planks（JSON 字符串，方便必要时重现）

说明：
- 不再控制主启发式 HV_primary 或 SU，仅针对“挡板数量次启发式”做设计。
- 可以配合一个专门的运行脚本或在现有 runtime 中单独读这个 CSV 来做次启发式实验。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    BALL_START_Y,
    CATCHER_LEFT_X,
    CATCHER_RIGHT_X,
    N_PLANKS,
)
from physics_utils import (
    create_space,
    create_ball,
    create_plank,
    create_catchers,
    generate_random_planks,
    run_simulation,
    count_planks_per_side,
)

# 设计参数（可根据需要手动调整）
TRIALS_PER_CELL = 30  # 每个( HV_count × CountHeur_side )组合的目标试次数
PLANK_DIFF_MIN = 4    # 左右挡板数量差的最小绝对值 |N_left - N_right|
MAX_ATTEMPTS_PER_CELL = 8000  # 每个组合允许的最大尝试次数总数
ATTEMPTS_PER_CALL = 400       # 每次调用内部生成函数的最大尝试数


def generate_board_for_count_condition(target_hv_count: str,
                                       target_more_side: str,
                                       min_diff: int = PLANK_DIFF_MIN,
                                       max_attempts: int = 1000):
    """为指定的( HV_count, CountHeur_side )组合生成一块板配置。

    参数：
    - target_hv_count: 'congruent' 或 'conflict'
    - target_more_side: 'left' 或 'right'（挡板数量更多的一侧）
    - min_diff: 左右挡板数量的最小差值 |N_left - N_right|
    - max_attempts: 在本次调用中的最多尝试次数

    返回：dict 或 None
    dict 包含：planks, trajectory, phys_outcome, ball_start_side, ball_start_x,
              left_plank_count, right_plank_count, plank_diff, more_plank_side, hv_count
    """
    assert target_hv_count in ("congruent", "conflict")
    assert target_more_side in ("left", "right")

    attempts = 0

    last_planks = None
    last_trajectory = None
    last_start_side = "left"
    last_outcome = "none"

    while attempts < max_attempts:
        attempts += 1

        # 1. 随机生成挡板
        planks = generate_random_planks(N_PLANKS, seed=None)

        # 2. 统计左右挡板数量，要求差异足够大
        left_count, right_count = count_planks_per_side(planks)
        if left_count == right_count:
            continue
        diff = left_count - right_count
        if abs(diff) < min_diff:
            continue

        more_side = "left" if diff > 0 else "right"
        if more_side != target_more_side:
            continue

        # 3. 随机决定球的起点侧（与挡板数量启发式无直接关系）
        start_side = np.random.choice(["left", "right"])
        start_x = CATCHER_LEFT_X if start_side == "left" else CATCHER_RIGHT_X

        # 4. 物理模拟
        space = create_space()
        ball_body, ball_shape = create_ball(space, start_x, BALL_START_Y)
        create_catchers(space)
        for p in planks:
            create_plank(space, p["x"], p["y"], p["angle"])

        outcome, trajectory, _ = run_simulation(space, ball_body, ball_shape, planks)

        # 只保留真正落入左右接球器的 trial
        if outcome not in ("left", "right"):
            # 记录最后一次结果，万一整个 cell 都很难生成时可用于诊断
            last_planks = planks
            last_trajectory = trajectory
            last_start_side = start_side
            last_outcome = outcome
            continue

        count_heur_side = more_side  # 挡板数量启发式预测侧 = 挡板更多的一侧

        # 5. 根据挡板数量启发式与真实结果的关系筛选 HV_count
        if target_hv_count == "congruent" and outcome != count_heur_side:
            last_planks = planks
            last_trajectory = trajectory
            last_start_side = start_side
            last_outcome = outcome
            continue

        if target_hv_count == "conflict" and outcome == count_heur_side:
            last_planks = planks
            last_trajectory = trajectory
            last_start_side = start_side
            last_outcome = outcome
            continue

        # 满足所有条件，返回该板
        return {
            "planks": planks,
            "trajectory": trajectory,
            "phys_outcome": outcome,
            "ball_start_side": start_side,
            "ball_start_x": start_x,
            "left_plank_count": left_count,
            "right_plank_count": right_count,
            "plank_diff": diff,
            "more_plank_side": more_side,
            "hv_count": target_hv_count,
        }

    # 如果在 max_attempts 内都没找到，就返回 None（外层负责统计）
    return None


def generate_all_stimuli():
    """生成所有基于挡板数量次启发式的刺激板。"""
    cells = []
    for hv_count in ["congruent", "conflict"]:
        for more_side in ["left", "right"]:
            cells.append((hv_count, more_side))

    all_boards = []

    print("=" * 60)
    print("基于挡板数量次启发式的刺激生成")
    print("目标：每个 (HV_count × CountHeur_side) 组合各 {} 个，共 {} 个".format(
        TRIALS_PER_CELL, TRIALS_PER_CELL * len(cells)
    ))
    print("挡板数量差异阈值 |N_left - N_right| ≥ {}".format(PLANK_DIFF_MIN))
    print("=" * 60)

    for hv_count, more_side in cells:
        need = TRIALS_PER_CELL
        attempts_left = MAX_ATTEMPTS_PER_CELL
        generated_here = 0

        print(f"\n组合: HV_count={hv_count}, CountHeur_side={more_side}")
        print(f"  目标 {TRIALS_PER_CELL} 个，本轮最多尝试 {attempts_left} 次")

        while need > 0 and attempts_left > 0:
            this_attempts = min(ATTEMPTS_PER_CALL, attempts_left)
            board = generate_board_for_count_condition(
                hv_count,
                more_side,
                min_diff=PLANK_DIFF_MIN,
                max_attempts=this_attempts,
            )
            attempts_left -= this_attempts

            if board is None:
                # 当前这批尝试未找到合适板，继续下一批
                print(
                    f"    已尝试 {MAX_ATTEMPTS_PER_CELL - attempts_left}/{MAX_ATTEMPTS_PER_CELL} 次，"
                    f"当前成功 {generated_here} 个"
                )
                continue

            all_boards.append(board)
            generated_here += 1
            need -= 1

            if generated_here % 5 == 0 or need == 0:
                print(
                    f"    已成功生成 {generated_here} 个，剩余 {need} 个，"
                    f"总尝试 {MAX_ATTEMPTS_PER_CELL - attempts_left}/{MAX_ATTEMPTS_PER_CELL}"
                )

        print(
            f"  -> 组合 HV_count={hv_count}, CountHeur_side={more_side} 最终生成 {generated_here} 个，"
            f"仍缺 {need} 个"
        )

    return all_boards


def save_stimuli(boards, output_dir: str = "stimuli") -> Path:
    """将生成的板配置保存为 CSV 文件。"""
    script_dir = Path(__file__).parent
    out_dir = script_dir / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, b in enumerate(boards):
        record = {
            "BoardID": f"B{idx:04d}",
            "HV_count": b["hv_count"],
            "CountHeur_side": b["more_plank_side"],
            "N_planks_left": b["left_plank_count"],
            "N_planks_right": b["right_plank_count"],
            "PlankCount_diff": int(b["plank_diff"]),
            "PhysOutcome": b["phys_outcome"],
            "Ball_start_side": b["ball_start_side"],
            "Ball_start_x": float(b["ball_start_x"]),
            "Ball_start_y": BALL_START_Y,
            "True_trajectory": json.dumps(b["trajectory"]),
            "Planks": json.dumps(b["planks"]),
        }
        records.append(record)

    df = pd.DataFrame(records)

    out_path = out_dir / "stimuli_count_heuristic.csv"
    df.to_csv(out_path, index=False)

    print("\n刺激已保存到:", out_path)

    # 打印简单分布统计，便于检查设计
    print("\n" + "=" * 60)
    print("刺激分布统计 (基于挡板数量启发式)")
    print("=" * 60)
    print("\nHV_count 分布:")
    print(df["HV_count"].value_counts())

    print("\nCountHeur_side 分布:")
    print(df["CountHeur_side"].value_counts())

    print("\nHV_count × CountHeur_side 组合分布:")
    print(df.groupby(["HV_count", "CountHeur_side"]).size().unstack(fill_value=0))

    print("\n挡板数量差值 |N_left - N_right| 的描述统计:")
    print(df["PlankCount_diff"].abs().describe())

    return out_path


if __name__ == "__main__":
    boards = generate_all_stimuli()
    save_stimuli(boards)
