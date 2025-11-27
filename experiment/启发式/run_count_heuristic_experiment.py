from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from psychopy import visual, event, core, gui

# 确保可以从脚本所在目录导入 config 等模块
SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    FPS,
    BALL_RADIUS,
    BALL_START_Y,
    CATCHER_LEFT_X,
    CATCHER_RIGHT_X,
    CATCHER_Y,
    CATCHER_WIDTH,
    CATCHER_HEIGHT,
    PLANK_WIDTH,
    PLANK_HEIGHT,
    FIXATION_DURATION,
    FIXATION_JITTER,
)


VISUAL_DT = 1.0 / FPS


def phys_to_psychopy(x: float, y: float):
    """将物理坐标(左上为原点, y向下)转换为PsychoPy像素坐标(屏幕中心为原点, y向上)。"""
    x_pix = x - SCREEN_WIDTH / 2.0
    y_pix = (SCREEN_HEIGHT / 2.0) - y
    return x_pix, y_pix


def load_stimuli(n_trials: int = 60) -> pd.DataFrame:
    """从 stimuli_count_heuristic.csv 读取刺激，并选出本被试要呈现的试次。"""
    # 优先从子文件夹 stimuli 读取；若不存在，则回退到脚本同一文件夹
    stim_path = SCRIPT_DIR / "stimuli" / "stimuli_count_heuristic.csv"
    if not stim_path.exists():
        alt_path = SCRIPT_DIR / "stimuli_count_heuristic.csv"
        if alt_path.exists():
            stim_path = alt_path
        else:
            raise FileNotFoundError(
                f"找不到刺激文件: {stim_path} 或 {alt_path}\n"
                f"请先运行 generate_count_heuristic_stimuli.py 生成，"
                f"或确认 CSV 放在与本脚本同一文件夹。"
            )

    df = pd.read_csv(stim_path)

    if len(df) <= n_trials:
        selected = df.sample(frac=1).reset_index(drop=True)
    else:
        # 先尽量在 4 个组合 (HV_count × CountHeur_side) 之间平均抽样
        target_per_cell = n_trials // 4
        selected_indices = []
        remaining = n_trials

        grouped = df.groupby(["HV_count", "CountHeur_side"])
        for (hv, side), group in grouped:
            if remaining <= 0:
                break
            take = min(target_per_cell, len(group))
            if take <= 0:
                continue
            idx = np.random.choice(group.index, size=take, replace=False)
            selected_indices.extend(idx.tolist())
            remaining -= take

        # 如果还不够，再从剩余 trial 中补足
        if remaining > 0:
            remaining_pool = df.index.difference(selected_indices)
            if len(remaining_pool) >= remaining:
                extra = np.random.choice(remaining_pool, size=remaining, replace=False)
            else:
                # 极端情况：总 trial 数也不多，允许有放回抽样
                extra = np.random.choice(df.index, size=remaining, replace=True)
            selected_indices.extend(extra.tolist())

        selected = df.loc[selected_indices].sample(frac=1).reset_index(drop=True)

    # 解析 JSON 列
    selected["Planks_obj"] = selected["Planks"].apply(json.loads)
    selected["Trajectory_obj"] = selected["True_trajectory"].apply(json.loads)

    return selected


def create_board_visuals(win, planks):
    """根据挡板配置在 PsychoPy 中创建静态刺激对象。"""
    plank_stimuli = []
    for p in planks:
        x_pix, y_pix = phys_to_psychopy(p["x"], p["y"])
        stim = visual.Rect(
            win,
            width=PLANK_WIDTH,
            height=PLANK_HEIGHT,
            pos=(x_pix, y_pix),
            ori=-p["angle"],
            fillColor="white",
            lineColor="white",
            units="pix",
        )
        plank_stimuli.append(stim)

    # 接球器底部
    left_x_pix, left_y_pix = phys_to_psychopy(CATCHER_LEFT_X, CATCHER_Y)
    right_x_pix, right_y_pix = phys_to_psychopy(CATCHER_RIGHT_X, CATCHER_Y)

    left_catcher = visual.Rect(
        win,
        width=CATCHER_WIDTH,
        height=CATCHER_HEIGHT,
        pos=(left_x_pix, left_y_pix),
        fillColor="white",
        lineColor="white",
        units="pix",
    )
    right_catcher = visual.Rect(
        win,
        width=CATCHER_WIDTH,
        height=CATCHER_HEIGHT,
        pos=(right_x_pix, right_y_pix),
        fillColor="white",
        lineColor="white",
        units="pix",
    )

    # 接球器侧壁（与 physics_utils.create_catchers 中的几何一致）
    wall_height = 80
    wall_width = 10
    wall_center_y = CATCHER_Y - wall_height / 2.0

    left_wall_left_x = CATCHER_LEFT_X - CATCHER_WIDTH / 2.0 + wall_width / 2.0
    left_wall_right_x = CATCHER_LEFT_X + CATCHER_WIDTH / 2.0 - wall_width / 2.0
    right_wall_left_x = CATCHER_RIGHT_X - CATCHER_WIDTH / 2.0 + wall_width / 2.0
    right_wall_right_x = CATCHER_RIGHT_X + CATCHER_WIDTH / 2.0 - wall_width / 2.0

    for x_phys in (left_wall_left_x, left_wall_right_x, right_wall_left_x, right_wall_right_x):
        wx_pix, wy_pix = phys_to_psychopy(x_phys, wall_center_y)
        wall = visual.Rect(
            win,
            width=wall_width,
            height=wall_height,
            pos=(wx_pix, wy_pix),
            fillColor="white",
            lineColor="white",
            units="pix",
        )
        plank_stimuli.append(wall)

    # 起始小球位置（根据 Ball_start_x / BALL_START_Y 动态设置）
    ball = visual.Circle(
        win,
        radius=BALL_RADIUS,
        pos=(0, 0),
        fillColor="white",
        lineColor="white",
        units="pix",
    )

    return plank_stimuli, left_catcher, right_catcher, ball


def show_instructions(win, n_trials: int):
    text = (
        "挡板数量启发式实验\n\n"
        "屏幕上会出现一个装置，上方有一个小球，下面有左右两个接球器。\n\n"
        "你的任务是：在球下落之前，预测它最终会落入左边还是右边的接球器。\n\n"
        "本实验只需要二选一，不需要输入信心度：\n"
        "全程使用鼠标操作 —— 用鼠标点击你认为小球会落入的接球器（左/右）。\n\n"
        f"本实验共 {n_trials} 个试次，中途如需终止，可以按 ESC 键退出。\n\n"
        "按空格键开始实验，或按 ESC 退出。"
    )
    stim = visual.TextStim(
        win,
        text=text,
        height=28,
        color="white",
        wrapWidth=SCREEN_WIDTH * 0.9,
        units="pix",
    )
    stim.draw()
    win.flip()

    keys = event.waitKeys(keyList=["space", "escape"])
    if "escape" in keys:
        win.close()
        core.quit()


def show_fixation(win):
    """呈现注视十字，时长 = FIXATION_DURATION ± FIXATION_JITTER。"""
    dur_ms = FIXATION_DURATION + np.random.uniform(-FIXATION_JITTER, FIXATION_JITTER)
    dur_s = max(0.1, dur_ms / 1000.0)

    stim = visual.TextStim(
        win,
        text="+",
        height=40,
        color="white",
        units="pix",
    )
    stim.draw()
    win.flip()
    core.wait(dur_s)


def collect_choice(win, trial_index: int, n_trials: int, board_visuals, ball_start_x: float):
    """显示当前板子并采集被试的二选一反应（鼠标点击左右接球器）。返回 response_side, RT。"""
    planks, left_catcher, right_catcher, ball = board_visuals

    # 起始球位置
    ball_x_pix, ball_y_pix = phys_to_psychopy(ball_start_x, BALL_START_Y)
    ball.pos = (ball_x_pix, ball_y_pix)

    # 鼠标用于左右选择（左键=左边，右键=右边），不再限制点击位置
    mouse = event.Mouse(win=win, visible=True)
    prev_left = False
    prev_right = False

    clock = core.Clock()
    event.clearEvents()
    clock.reset()

    while True:
        for p in planks:
            p.draw()
        left_catcher.draw()
        right_catcher.draw()
        ball.draw()
        win.flip()

        # 鼠标左/右键从未按下 -> 按下时，根据按键直接决定左右
        left, _, right = mouse.getPressed()

        # 左键第一次按下 → 选择左边
        if left and not prev_left:
            rt = float(clock.getTime())
            return "left", rt

        # 右键第一次按下 → 选择右边
        if right and not prev_right:
            rt = float(clock.getTime())
            return "right", rt

        prev_left = left
        prev_right = right

        # 仍然允许 ESC 中止实验
        if "escape" in event.getKeys(keyList=["escape"]):
            win.close()
            core.quit()


def play_trajectory(win, trajectory, board_visuals):
    """沿着预先计算好的轨迹播放下落动画。"""
    planks, left_catcher, right_catcher, ball = board_visuals

    for point in trajectory:
        x_pix, y_pix = phys_to_psychopy(point["x"], point["y"])
        ball.pos = (x_pix, y_pix)

        for p in planks:
            p.draw()
        left_catcher.draw()
        right_catcher.draw()
        ball.draw()
        win.flip()

        if "escape" in event.getKeys(keyList=["escape"]):
            win.close()
            core.quit()

        core.wait(VISUAL_DT)


def run_experiment(subject_id: str, n_trials: int = 60):
    stimuli = load_stimuli(n_trials=n_trials)

    win = visual.Window(
        size=(SCREEN_WIDTH, SCREEN_HEIGHT),
        units="pix",
        fullscr=False,
        color=[0, 0, 0],
    )

    show_instructions(win, n_trials=n_trials)

    records = []

    for i, row in stimuli.iterrows():
        show_fixation(win)

        planks = row["Planks_obj"]
        trajectory = row["Trajectory_obj"]
        board_visuals = create_board_visuals(win, planks)

        response_side, rt = collect_choice(
            win,
            trial_index=i,
            n_trials=n_trials,
            board_visuals=board_visuals,
            ball_start_x=row["Ball_start_x"],
        )

        play_trajectory(win, trajectory, board_visuals)

        correct_side = row["PhysOutcome"]
        correct = int(correct_side == response_side)

        record = {
            "SubjectID": subject_id,
            "TrialIndex": int(i),
            "BoardID": row["BoardID"],
            "HV_count": row["HV_count"],
            "CountHeur_side": row["CountHeur_side"],
            "N_planks_left": int(row["N_planks_left"]),
            "N_planks_right": int(row["N_planks_right"]),
            "PlankCount_diff": int(row["PlankCount_diff"]),
            "PhysOutcome": correct_side,
            "Ball_start_side": row["Ball_start_side"],
            "Response": response_side,
            "Correct": correct,
            "RT": rt,
        }
        records.append(record)

    # 结束界面
    end_text = visual.TextStim(
        win,
        text="实验结束，感谢参与！\n\n请通知实验员。\n\n按任意键退出。",
        height=28,
        color="white",
        wrapWidth=SCREEN_WIDTH * 0.9,
        units="pix",
    )
    end_text.draw()
    win.flip()
    event.waitKeys()
    win.close()

    # 保存数据
    out_dir = SCRIPT_DIR / "data_count_heuristic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"behavioral_count_heuristic_{subject_id}.csv"
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_path, index=False)
    print("行为数据已保存到:", out_path)


def main():
    dlg = gui.Dlg(title="挡板数量启发式实验")
    dlg.addText("请输入被试信息：")
    dlg.addField("被试编号 (如 S01):", "S01")
    ok_data = dlg.show()
    if not dlg.OK:
        core.quit()

    subject_id = str(ok_data[0]).strip() or "NA"
    n_trials = 60
    run_experiment(subject_id=subject_id, n_trials=n_trials)


if __name__ == "__main__":
    main()
