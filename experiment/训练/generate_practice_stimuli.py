"""\
训练用练习刺激生成脚本

作用：
- 在当前目录下生成一个 stimuli_practice.csv
- 里面包含若干随机 trial，每个 trial 一块板：
  * 随机挡板（不做 SU/HV 约束）
  * 随机起点在左/右杯正上方
  * 球真实模拟后必须落入左右任一接球器
- 供 训练/practice_demo.py 在 PsychoPy 中读取并做练习使用。

注意：
- 依赖 physics_utils 和 pymunk，只在你本机装好 pymunk 的 Python 环境里运行一次即可。
- 练习运行脚本本身不依赖 pymunk，只读取这个 CSV。
"""

from pathlib import Path
import sys
import json
from typing import List, Dict

import numpy as np
import pandas as pd

# 保证能从本目录导入 config 和 physics_utils
SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    BALL_START_Y,
    CATCHER_LEFT_X,
    CATCHER_RIGHT_X,
    CATCHER_Y,
    N_PLANKS,
)
from physics_utils import (
    create_space,
    create_ball,
    create_catchers,
    create_plank,
    generate_random_planks,
    run_simulation,
    check_straight_line_heuristic,
)


N_PRACTICE_STIMULI = 100  # 目标练习 trial 数
MAX_ATTEMPTS = 5000       # 全局最大尝试次数（防止无限循环）


def generate_one_trial() -> Dict:
    """生成一个随机 trial，要求球最终落入左右任一接球器。

    返回包含 planks, trajectory, outcome, start_side, start_x 的 dict；
    若当前尝试失败（outcome=='none'），返回 None。
    """
    planks = generate_random_planks(N_PLANKS, seed=None)

    start_side = np.random.choice(["left", "right"])
    start_x = CATCHER_LEFT_X if start_side == "left" else CATCHER_RIGHT_X

    # 要求练习trial在起点到对应接球器的直线路径上必须有挡板
    straight_clear = check_straight_line_heuristic(
        start_x,
        BALL_START_Y,
        start_x,
        CATCHER_Y,
        planks,
    )
    if straight_clear:
        return None

    space = create_space()
    ball_body, ball_shape = create_ball(space, start_x, BALL_START_Y)
    create_catchers(space)
    for p in planks:
        create_plank(space, p["x"], p["y"], p["angle"])

    outcome, trajectory, _ = run_simulation(space, ball_body, ball_shape, planks)
    if outcome not in ("left", "right") or not trajectory:
        return None

    return {
        "planks": planks,
        "trajectory": trajectory,
        "phys_outcome": outcome,
        "ball_start_side": start_side,
        "ball_start_x": float(start_x),
    }


def generate_all() -> List[Dict]:
    boards: List[Dict] = []
    attempts = 0

    print("=" * 60)
    print("训练用随机练习刺激生成")
    print(f"目标 trial 数: {N_PRACTICE_STIMULI}")
    print("不控制 SU / HV，仅要求球落入左右接球器。")
    print("=" * 60)

    while len(boards) < N_PRACTICE_STIMULI and attempts < MAX_ATTEMPTS:
        attempts += 1
        board = generate_one_trial()
        if board is None:
            continue
        boards.append(board)
        if len(boards) % 10 == 0 or len(boards) == N_PRACTICE_STIMULI:
            print(f"  已生成 {len(boards)}/{N_PRACTICE_STIMULI} 个有效 trial (尝试 {attempts} 次)")

    print(f"\n最终生成 {len(boards)} 个有效 trial, 总尝试 {attempts} 次。")
    if len(boards) < N_PRACTICE_STIMULI:
        print("⚠ 未达到目标数量，但仍可用于练习。")

    return boards


def save_to_csv(boards: List[Dict]) -> Path:
    out_path = SCRIPT_DIR / "stimuli_practice.csv"

    records = []
    for i, b in enumerate(boards):
        records.append(
            {
                "BoardID": f"P{i:04d}",
                "PhysOutcome": b["phys_outcome"],
                "Ball_start_side": b["ball_start_side"],
                "Ball_start_x": b["ball_start_x"],
                "Ball_start_y": BALL_START_Y,
                "True_trajectory": json.dumps(b["trajectory"]),
                "Planks": json.dumps(b["planks"]),
            }
        )

    df = pd.DataFrame.from_records(records)
    df.to_csv(out_path, index=False)

    print("\n练习刺激已保存到:", out_path)
    print("总试次:", len(df))
    print("PhysOutcome 分布:")
    print(df["PhysOutcome"].value_counts())

    return out_path


if __name__ == "__main__":
    boards = generate_all()
    save_to_csv(boards)
