from pathlib import Path
import sys
import csv
import json
import random
from typing import List, Dict

import numpy as np
from psychopy import visual, event, core

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
    N_PLANKS,
)


VISUAL_DT = 1.0 / FPS
N_PRACTICE_TRIALS = 20  # 每个被试实际练习的trial数

STIMULI_PATH = SCRIPT_DIR / "stimuli_practice.csv"
_PRACTICE_CACHE: List[Dict] | None = None


def load_practice_stimuli() -> List[Dict]:
    """从 stimuli_practice.csv 读取练习刺激并解析 JSON 字段。"""
    global _PRACTICE_CACHE
    if _PRACTICE_CACHE is not None:
        return _PRACTICE_CACHE

    if not STIMULI_PATH.exists():
        raise FileNotFoundError(
            f"找不到练习刺激文件: {STIMULI_PATH}\n请先运行 generate_practice_stimuli.py 生成。"
        )

    with STIMULI_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError("练习刺激文件 stimuli_practice.csv 为空。")

    for r in rows:
        try:
            r["Planks_obj"] = json.loads(r["Planks"])
            r["Trajectory_obj"] = json.loads(r["True_trajectory"])
        except Exception:
            r["Planks_obj"] = None
            r["Trajectory_obj"] = None

    _PRACTICE_CACHE = rows
    return rows

def phys_to_psychopy(x: float, y: float):
    """将物理坐标(左上为原点, y向下)转换为PsychoPy像素坐标(屏幕中心为原点, y向上)。"""
    x_pix = x - SCREEN_WIDTH / 2.0
    y_pix = (SCREEN_HEIGHT / 2.0) - y
    return x_pix, y_pix


def create_board_visuals(win, planks, start_side):
    """根据挡板配置和起点，在PsychoPy中创建静态刺激对象。"""
    plank_stimuli = []
    for p in planks:
        x_pix, y_pix = phys_to_psychopy(p["x"], p["y"])
        stim = visual.Rect(
            win,
            width=PLANK_WIDTH,
            height=PLANK_HEIGHT,
            pos=(x_pix, y_pix),
            ori=-p["angle"],  # 角度方向和Pymunk略有差异，这里取相反号近似
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
    wall_height = 80
    wall_width = 10
    wall_center_y = CATCHER_Y - wall_height / 2.0

    left_wall_left_x = CATCHER_LEFT_X - CATCHER_WIDTH / 2.0 + wall_width / 2.0
    left_wall_right_x = CATCHER_LEFT_X + CATCHER_WIDTH / 2.0 - wall_width / 2.0
    right_wall_left_x = CATCHER_RIGHT_X - CATCHER_WIDTH / 2.0 + wall_width / 2.0
    right_wall_right_x = CATCHER_RIGHT_X + CATCHER_WIDTH / 2.0 - wall_width / 2.0

    for x_phys in (left_wall_left_x, left_wall_right_x, right_wall_left_x, right_wall_right_x):
        x_pix, y_pix = phys_to_psychopy(x_phys, wall_center_y)
        wall = visual.Rect(
            win,
            width=wall_width,
            height=wall_height,
            pos=(x_pix, y_pix),
            fillColor="white",
            lineColor="white",
            units="pix",
        )
        plank_stimuli.append(wall)

    # 起始小球
    start_x = CATCHER_LEFT_X if start_side == "left" else CATCHER_RIGHT_X
    ball_x_pix, ball_y_pix = phys_to_psychopy(start_x, BALL_START_Y)
    ball = visual.Circle(
        win,
        radius=BALL_RADIUS,
        pos=(ball_x_pix, ball_y_pix),
        fillColor="white",
        lineColor="white",
        units="pix",
    )

    return plank_stimuli, left_catcher, right_catcher, ball


def show_instructions(win):
    """显示练习阶段的总体指导语。"""
    text = (
        "练习阶段\n\n"
        "屏幕上会出现一个装置，上方有一个小球，下面有左右两个接球器。\n\n"
        "你的任务是：在球下落之前，预测它最终会落入左边还是右边的接球器。\n\n"
        "正式实验中，你将按 F 键表示‘左边’，按 J 键表示‘右边’。\n\n"
        "在接下来的练习中，请先做出预测，然后会播放球的真实轨迹供你观察。\n\n"
        "本练习脚本不会记录任何数据，只是帮助你理解任务。\n\n"
        "按空格键开始练习，或按 ESC 退出。"
    )
    stim = visual.TextStim(
        win,
        text=text,
        height=32,
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


def show_prediction_screen(win, trial_index, n_trials, board_visuals):
    """显示当前板子，请被试做出预测（F=左, J=右）。"""
    planks, left_catcher, right_catcher, ball = board_visuals
    while True:
        for p in planks:
            p.draw()
        left_catcher.draw()
        right_catcher.draw()
        ball.draw()
        win.flip()

        keys = event.getKeys(keyList=["f", "j", "escape"])
        if "escape" in keys:
            win.close()
            core.quit()
        if "f" in keys:
            return "left"
        if "j" in keys:
            return "right"


def play_trajectory(win, trajectory, board_visuals):
    """播放球的真实下落轨迹动画。"""
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


def show_feedback(win, predicted_side, outcome):
    """根据被试预测和真实落点给出文字反馈，不记录到任何文件。"""
    if outcome not in ("left", "right"):
        text = "本次模拟中小球没有落入接球器，这种情况在正式实验中不会出现。\n\n按空格键继续。"
    else:
        correct = (predicted_side == outcome)
        side_zh = "左边" if outcome == "left" else "右边"
        if correct:
            text = f"真实结果：小球落在{side_zh}接球器。\n你的判断是正确的！\n\n按空格键继续。"
        else:
            text = f"真实结果：小球落在{side_zh}接球器。\n这次判断与真实结果不一致。\n\n按空格键继续。"

    stim = visual.TextStim(
        win,
        text=text,
        height=32,
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


def run_practice(win, trials: List[Dict]):
    """运行若干个练习试次，仅用于理解任务，不保存任何数据。

    参数 trials: trial 字典列表（来自 stimuli_practice.csv 的若干行）。"""
    n_trials = len(trials)
    for i, row in enumerate(trials):
        planks = row["Planks_obj"]
        traj = row["Trajectory_obj"]
        start_side = row.get("Ball_start_side", "left")
        outcome = row.get("PhysOutcome", "none")

        if not planks or not traj or outcome not in ("left", "right"):
            continue

        board_visuals = create_board_visuals(win, planks, start_side)
        predicted = show_prediction_screen(win, i, n_trials, board_visuals)
        play_trajectory(win, traj, board_visuals)
        show_feedback(win, predicted, outcome)


def show_end_screen(win):
    text = (
        "练习结束。\n\n"
        "现在你已经大致了解了任务流程。\n\n"
        "接下来可以关闭本窗口，或者按 ESC 退出，然后由实验员启动正式实验脚本。"
    )
    stim = visual.TextStim(
        win,
        text=text,
        height=32,
        color="white",
        wrapWidth=SCREEN_WIDTH * 0.9,
        units="pix",
    )
    stim.draw()
    win.flip()

    keys = event.waitKeys(keyList=["escape", "space"])
    # 无论按空格还是 ESC，都直接退出
    win.close()
    core.quit()


def main():
    win = visual.Window(
        size=(SCREEN_WIDTH, SCREEN_HEIGHT),
        units="pix",
        fullscr=False,
        color=[0, 0, 0],
    )

    show_instructions(win)

    rows = load_practice_stimuli()
    # 只保留解析成功且 outcome 合法的 trial
    valid = [r for r in rows if r.get("Planks_obj") and r.get("Trajectory_obj") and r.get("PhysOutcome") in ("left", "right")]
    if not valid:
        raise RuntimeError("练习刺激文件中没有可用 trial。")

    if len(valid) <= N_PRACTICE_TRIALS:
        selected = valid
    else:
        indices = np.random.choice(len(valid), size=N_PRACTICE_TRIALS, replace=False)
        selected = [valid[i] for i in indices]

    run_practice(win, selected)
    show_end_screen(win)


if __name__ == "__main__":
    main()
