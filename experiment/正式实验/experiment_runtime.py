"""
实验运行脚本
使用PsychoPy呈现刺激、收集数据、同步Eyelink眼动仪
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox

try:
    import pylink
except Exception:
    pylink = None
    print("Warning: 'pylink' is not available; Eyelink recording will be disabled.")

from psychopy import visual, core, event, gui, monitors
from psychopy.hardware import keyboard

from config import *
import config  # 也需要import模块本身以便调用update_screen_params

# 脚本所在目录，用于构造刺激文件等资源的绝对路径
SCRIPT_DIR = Path(__file__).parent


class SubjectInfoDialog:
    """被试信息收集对话框"""
    
    def __init__(self):
        self.info = None
        
    def show(self):
        """显示对话框并返回被试信息"""
        root = tk.Tk()
        root.title("被试信息")
        root.geometry("400x300")
        root.resizable(False, False)
        
        # 居中显示
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (400 // 2)
        y = (root.winfo_screenheight() // 2) - (300 // 2)
        root.geometry(f"400x300+{x}+{y}")
        
        # 创建输入框
        fields = {}
        
        # 被试ID
        tk.Label(root, text="被试ID:", font=("Arial", 12)).grid(row=0, column=0, padx=20, pady=15, sticky='e')
        fields['id'] = tk.Entry(root, font=("Arial", 12), width=20)
        fields['id'].grid(row=0, column=1, padx=20, pady=15)
        
        # 性别
        tk.Label(root, text="性别:", font=("Arial", 12)).grid(row=1, column=0, padx=20, pady=15, sticky='e')
        fields['gender'] = ttk.Combobox(root, font=("Arial", 12), width=18, state='readonly')
        fields['gender']['values'] = ('男', '女', '其他')
        fields['gender'].current(0)
        fields['gender'].grid(row=1, column=1, padx=20, pady=15)
        
        # 年龄
        tk.Label(root, text="年龄:", font=("Arial", 12)).grid(row=2, column=0, padx=20, pady=15, sticky='e')
        fields['age'] = tk.Entry(root, font=("Arial", 12), width=20)
        fields['age'].grid(row=2, column=1, padx=20, pady=15)
        
        # 球类经验（0-100自评分）
        tk.Label(root, text="球类经验 (0-100):", font=("Arial", 12)).grid(row=3, column=0, padx=20, pady=15, sticky='e')
        
        # 创建框架容纳滑块和数值显示
        exp_frame = tk.Frame(root)
        exp_frame.grid(row=3, column=1, padx=20, pady=15)
        
        fields['ball_exp_var'] = tk.IntVar(value=50)
        fields['ball_exp'] = tk.Scale(exp_frame, from_=0, to=100, orient='horizontal',
                                       variable=fields['ball_exp_var'], length=150, 
                                       font=("Arial", 10))
        fields['ball_exp'].pack(side='left')
        
        # 显示当前数值
        fields['ball_exp_label'] = tk.Label(exp_frame, textvariable=fields['ball_exp_var'], 
                                            font=("Arial", 12), width=3)
        fields['ball_exp_label'].pack(side='left', padx=5)
        
        # 确认按钮
        def on_confirm():
            # 验证输入
            if not fields['id'].get().strip():
                messagebox.showerror("错误", "请输入被试ID")
                return
            
            try:
                age = int(fields['age'].get().strip())
                if age < 0 or age > 120:
                    raise ValueError
            except ValueError:
                messagebox.showerror("错误", "请输入有效的年龄（0-120）")
                return
            
            # 保存信息
            self.info = {
                'SubjectID': fields['id'].get().strip(),
                'Gender': fields['gender'].get(),
                'Age': age,
                'BallExperience': fields['ball_exp_var'].get()  # 0-100的数值
            }
            root.destroy()
        
        tk.Button(root, text="确认", font=("Arial", 12), width=10, 
                 command=on_confirm).grid(row=4, column=0, columnspan=2, pady=20)
        
        root.mainloop()
        
        return self.info


class PlankoExperiment:
    """Planko实验主类"""
    
    def __init__(self, subject_info, stimuli_file=None):
        self.subject_info = subject_info
        self.subject_id = subject_info['SubjectID']
        
        # 解析刺激文件路径：若未指定或为相对路径，则相对于脚本目录
        if stimuli_file is None:
            stimuli_path = SCRIPT_DIR / 'stimuli' / 'stimulus_config.csv'
        else:
            stimuli_path = Path(stimuli_file)
            if not stimuli_path.is_absolute():
                stimuli_path = SCRIPT_DIR / stimuli_path

        if not stimuli_path.exists():
            raise FileNotFoundError(f"找不到刺激文件: {stimuli_path}")

        # 加载刺激配置
        self.stimuli_df = pd.read_csv(stimuli_path)
        print(f"加载了 {len(self.stimuli_df)} 个刺激配置 (来自 {stimuli_path})")
        
        # 准备试次序列
        self.prepare_trials()
        
        # 初始化数据记录
        self.trial_data = []
        
        # 创建输出目录
        self.output_dir = Path('data') / f'sub-{self.subject_id}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化窗口和刺激（稍后在setup中完成）
        self.win = None
        self.kb = None
        # Eyelink 相关
        self.el = None
        self.edf_file = None
        
    def prepare_trials(self):
        """准备试次序列：随机打乱后平均分配到blocks"""
        print(f"总试次数: {len(self.stimuli_df)}")
        
        # 随机打乱所有试次
        shuffled_trials = self.stimuli_df.sample(frac=1).reset_index(drop=True)
        
        # 计算每个block的试次数
        trials_per_block = len(shuffled_trials) // N_BLOCKS
        
        # 分配到各个blocks
        blocks = []
        for block_id in range(1, N_BLOCKS + 1):
            start_idx = (block_id - 1) * trials_per_block
            end_idx = block_id * trials_per_block if block_id < N_BLOCKS else len(shuffled_trials)
            
            block_trials = shuffled_trials.iloc[start_idx:end_idx].copy()
            block_trials['BlockID'] = block_id
            block_trials['TrialID_in_block'] = range(1, len(block_trials) + 1)
            
            blocks.append(block_trials)
            print(f"Block {block_id}: {len(block_trials)} 个试次")
        
        # 合并所有blocks
        self.trial_sequence = pd.concat(blocks, ignore_index=True)
        self.trial_sequence['TrialID'] = range(1, len(self.trial_sequence) + 1)
        
        print(f"\n总计 {len(self.trial_sequence)} 个试次分配到 {N_BLOCKS} 个blocks")
        
    def setup_psychopy(self):
        """初始化PsychoPy窗口和刺激"""
        # 创建全屏窗口（不指定固定尺寸，让PsychoPy自动使用屏幕分辨率）
        self.win = visual.Window(
            fullscr=True,
            screen=0,
            allowGUI=False,
            monitor='testMonitor',
            color=[0, 0, 0],
            colorSpace='rgb',
            units='pix'
        )
        
        # 获取实际窗口尺寸并更新config参数
        actual_width, actual_height = self.win.size
        print(f"\n实际屏幕分辨率: {actual_width} x {actual_height}")
        print(f"设计分辨率: {DESIGN_WIDTH} x {DESIGN_HEIGHT}")
        
        # 更新所有config参数以适应实际屏幕
        config.update_screen_params(actual_width, actual_height)
        print(f"已更新参数以适应屏幕尺寸")
        
        # 计算坐标转换比例（用于将刺激配置从设计分辨率转换到实际分辨率）
        self.coord_scale_x = actual_width / DESIGN_WIDTH
        self.coord_scale_y = actual_height / DESIGN_HEIGHT
        
        # 创建键盘
        self.kb = keyboard.Keyboard()
        
        # 创建刺激对象（使用更新后的参数）
        self.fixation = visual.Circle(
            self.win, radius=BALL_RADIUS, fillColor='white', lineColor='white'
        )
        
        self.instruction_text = visual.TextStim(
            self.win, text='', 
            height=int(30 * min(self.coord_scale_x, self.coord_scale_y)), 
            color='white', 
            wrapWidth=int(1200 * self.coord_scale_x)
        )
        
        # 球和挡板（用于呈现静态场景和动画）
        self.ball = visual.Circle(
            self.win, radius=BALL_RADIUS, fillColor='red', lineColor='red'
        )
        
        # 接球器：视觉上画成“开口向上”的容器（底部 + 两侧壁）
        left_center_x = CATCHER_LEFT_X - SCREEN_WIDTH / 2
        right_center_x = CATCHER_RIGHT_X - SCREEN_WIDTH / 2
        catcher_y_psychopy = SCREEN_HEIGHT / 2 - CATCHER_Y

        # 底部条形（仍用原来的catcher_left / catcher_right 名称，方便后面使用）
        self.catcher_left = visual.Rect(
            self.win, width=CATCHER_WIDTH, height=CATCHER_HEIGHT,
            pos=(left_center_x, catcher_y_psychopy),
            fillColor='green', lineColor='green'
        )
        self.catcher_right = visual.Rect(
            self.win, width=CATCHER_WIDTH, height=CATCHER_HEIGHT,
            pos=(right_center_x, catcher_y_psychopy),
            fillColor='blue', lineColor='blue'
        )

        # 侧壁（窄而高的矩形），只用于视觉呈现
        wall_width = 10
        wall_height = 80

        self.catcher_left_parts = [self.catcher_left]
        self.catcher_left_parts.append(
            visual.Rect(
                self.win, width=wall_width, height=wall_height,
                pos=(left_center_x - CATCHER_WIDTH / 2 + wall_width / 2,
                     catcher_y_psychopy + wall_height / 2),
                fillColor='green', lineColor='green'
            )
        )
        self.catcher_left_parts.append(
            visual.Rect(
                self.win, width=wall_width, height=wall_height,
                pos=(left_center_x + CATCHER_WIDTH / 2 - wall_width / 2,
                     catcher_y_psychopy + wall_height / 2),
                fillColor='green', lineColor='green'
            )
        )

        self.catcher_right_parts = [self.catcher_right]
        self.catcher_right_parts.append(
            visual.Rect(
                self.win, width=wall_width, height=wall_height,
                pos=(right_center_x - CATCHER_WIDTH / 2 + wall_width / 2,
                     catcher_y_psychopy + wall_height / 2),
                fillColor='blue', lineColor='blue'
            )
        )
        self.catcher_right_parts.append(
            visual.Rect(
                self.win, width=wall_width, height=wall_height,
                pos=(right_center_x + CATCHER_WIDTH / 2 - wall_width / 2,
                     catcher_y_psychopy + wall_height / 2),
                fillColor='blue', lineColor='blue'
            )
        )
        
        # 信心度输入提示文字
        self.confidence_prompt = visual.TextStim(
            self.win, text='', 
            pos=(0, 50), height=40, color='white'
        )
        
        # 信心度当前值显示
        self.confidence_value_text = visual.TextStim(
            self.win, text='', 
            pos=(0, -100), height=80, color='yellow', bold=True
        )
        
        # 信心度提示文字
        self.confidence_instruction = visual.TextStim(
            self.win, text='', 
            pos=(0, -250), height=30, color='white'
        )
    
    def setup_eyelink(self):
        """初始化Eyelink（如果可用）"""
        if pylink is None:
            print("pylink 未安装，跳过Eyelink记录。")
            return

        try:
            self.el = pylink.EyeLink(None)
        except Exception as e:
            print(f"无法连接到Eyelink主机，跳过眼动记录: {e}")
            self.el = None
            return

        # 基于被试ID生成合法的EDF文件名（最多8个字母/数字）
        sid = str(self.subject_id)
        sid = "".join(ch for ch in sid if ch.isalnum()) or "S000"
        sid = sid[:8].upper()
        self.edf_file = f"{sid}.EDF"

        try:
            self.el.openDataFile(self.edf_file)
        except Exception as e:
            print(f"打开EDF文件失败，跳过眼动记录: {e}")
            try:
                self.el.close()
            except Exception:
                pass
            self.el = None
            return

        # 告诉Eyelink当前显示器的像素坐标
        try:
            self.el.sendCommand(
                "screen_pixel_coords 0 0 %d %d" % (SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1)
            )
            self.el.sendMessage(
                "DISPLAY_COORDS 0 0 %d %d" % (SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1)
            )
        except Exception as e:
            print(f"设置DISPLAY_COORDS失败: {e}")

        # 设置采样与事件过滤，确保EDF中包含视线位置、瞳孔面积、事件等
        try:
            self.el.setFileSampleFilter("LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS")
            self.el.setFileEventFilter(
                "LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON"
            )
            self.el.setLinkSampleFilter("LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS")
            self.el.setLinkEventFilter(
                "LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON"
            )
        except Exception as e:
            print(f"设置采样/事件过滤失败: {e}")

        # 定义静态兴趣区（AOI）：左右接球器和中央不确定区域（CU）
        try:
            wall_height = 80  # 与视觉呈现中的侧壁高度保持一致

            # 左接球器ROI（包含底部和侧壁）
            left_catcher_left = int(CATCHER_LEFT_X - CATCHER_WIDTH / 2)
            left_catcher_right = int(CATCHER_LEFT_X + CATCHER_WIDTH / 2)
            left_catcher_top = int(CATCHER_Y - wall_height)
            left_catcher_bottom = int(CATCHER_Y + CATCHER_HEIGHT / 2)
            self.el.sendMessage(
                f"!V IAREA RECTANGLE 1 {left_catcher_left} {left_catcher_top} "
                f"{left_catcher_right} {left_catcher_bottom} LEFT_CATCHER"
            )

            # 右接球器ROI
            right_catcher_left = int(CATCHER_RIGHT_X - CATCHER_WIDTH / 2)
            right_catcher_right = int(CATCHER_RIGHT_X + CATCHER_WIDTH / 2)
            right_catcher_top = int(CATCHER_Y - wall_height)
            right_catcher_bottom = int(CATCHER_Y + CATCHER_HEIGHT / 2)
            self.el.sendMessage(
                f"!V IAREA RECTANGLE 2 {right_catcher_left} {right_catcher_top} "
                f"{right_catcher_right} {right_catcher_bottom} RIGHT_CATCHER"
            )

            # 中央不确定区域（CU）：位于两接球器之间、挡板主要分布区域
            cu_left = int(CATCHER_LEFT_X + CATCHER_WIDTH / 2)
            cu_right = int(CATCHER_RIGHT_X - CATCHER_WIDTH / 2)
            cu_top = int(PLANK_Y_MIN)
            cu_bottom = int(CATCHER_Y - 20)
            if cu_right > cu_left and cu_bottom > cu_top:
                self.el.sendMessage(
                    f"!V IAREA RECTANGLE 3 {cu_left} {cu_top} "
                    f"{cu_right} {cu_bottom} CU_REGION"
                )
        except Exception as e:
            print(f"定义AOI失败: {e}")

        try:
            self.el.sendCommand(
                f"add_file_preamble_text 'SubjectID {self.subject_id}'"
            )
        except Exception:
            pass

        try:
            self.el.setOfflineMode()
            if pylink is not None:
                pylink.msecDelay(50)
        except Exception:
            pass

        print("Eyelink 已连接，EDF 文件名:", self.edf_file)
        print("请在Eyelink主机上完成校准/验证，然后开始实验。")
        
    def draw_planks(self, planks_json):
        """绘制挡板"""
        planks = json.loads(planks_json)
        plank_stims = []
        
        for p in planks:
            # 转换坐标系（Pymunk坐标 -> PsychoPy坐标）
            psychopy_x = p['x'] - SCREEN_WIDTH / 2
            psychopy_y = SCREEN_HEIGHT / 2 - p['y']
            
            plank = visual.Rect(
                self.win,
                width=PLANK_WIDTH,
                height=PLANK_HEIGHT,
                pos=(psychopy_x, psychopy_y),
                ori=-p['angle'],  # PsychoPy角度方向相反
                fillColor='white',
                lineColor='white'
            )
            plank_stims.append(plank)
        
        return plank_stims
    
    def show_instructions(self):
        """显示实验指导语"""
        instructions = """
        欢迎参加本实验！
        
        在每个试次中，您会看到一个游戏场景。
        每次小球将从左侧或右侧接球器正上方的固定位置下落（左右随机），
        如果碰到散落在屏幕中的挡板后将发生完全弹性碰撞，
        并最终落入左侧（绿色）或右侧（蓝色）的接球器。
        
        您的任务是：预测小球会落入哪个接球器。
        
        按键说明：
        - 按 'F' 键：预测小球落入左侧（绿色）
        - 按 'J' 键：预测小球落入右侧（蓝色）
        
        做出判断后，您需要报告对这个判断的信心度（0-100%）。
        
        然后您会观看到小球的真实下落动画和正确与否作为反馈。
        
        实验总共有5个block，每个block之间会有休息时间。
        
        准备好后，按空格键开始实验。
        """
        
        self.instruction_text.text = instructions
        self.instruction_text.draw()
        self.win.flip()
        
        # 等待空格键
        event.waitKeys(keyList=['space'])
    
    def run_trial(self, trial_info):
        """运行单个试次"""
        trial_record = {
            # 被试信息
            'SubjectID': self.subject_id,
            'Gender': self.subject_info['Gender'],
            'Age': self.subject_info['Age'],
            'BallExperience': self.subject_info['BallExperience'],
            # 试次信息
            'TrialID': trial_info['TrialID'],
            'TrialID_in_block': trial_info['TrialID_in_block'],
            'BlockID': trial_info['BlockID'],
            'BoardID': trial_info['BoardID'],
            # 刺激参数
            'SU_level': trial_info['SU_level'],
            'HV_primary': trial_info['HV_primary'],
            'HV_secondary_count': trial_info['HV_secondary_count'],
            'HV_secondary_collision': trial_info['HV_secondary_collision'],
            'PhysOutcome': trial_info['PhysOutcome'],
            'Ball_start_x': trial_info.get('Ball_start_x', BALL_START_X),
            'Ball_start_side': trial_info.get('Ball_start_side', '')
        }
        # 如果Eyelink可用，开始本trial记录并打TRIALID等标记
        if self.el is not None and pylink is not None:
            try:
                el = self.el
                el.setOfflineMode()
                pylink.msecDelay(5)

                trial_id = int(trial_info['TrialID'])
                block_id = int(trial_info['BlockID'])
                board_id = str(trial_info['BoardID'])
                hv = str(trial_info['HV_primary'])
                su_level = int(trial_info['SU_level'])
                outcome = str(trial_info['PhysOutcome'])

                el.sendMessage(
                    f"TRIALID {trial_id} BLOCK {block_id} BOARDID {board_id} "
                    f"HV {hv} SU {su_level} OUTCOME {outcome}"
                )
                # Data Viewer 变量标记
                el.sendMessage(f"!V TRIAL_VAR TrialID {trial_id}")
                el.sendMessage(f"!V TRIAL_VAR BlockID {block_id}")
                el.sendMessage(f"!V TRIAL_VAR BoardID {board_id}")
                el.sendMessage(f"!V TRIAL_VAR HV {hv}")
                el.sendMessage(f"!V TRIAL_VAR SU_level {su_level}")
                el.sendMessage(f"!V TRIAL_VAR Outcome {outcome}")

                el.startRecording(1, 1, 1, 1)
                pylink.msecDelay(20)
                el.sendMessage("TRIAL_START")
            except Exception as e:
                print(f"Eyelink trial start failed, continue without eye tracking: {e}")
        
        # 1. 注视点
        fixation_jitter = np.random.uniform(-FIXATION_JITTER, FIXATION_JITTER)
        fixation_duration = (FIXATION_DURATION + fixation_jitter) / 1000.0  # 转换为秒
        
        self.fixation.draw()
        self.win.flip()
        fixation_onset = core.getTime()
        trial_record['Fixation_onset'] = fixation_onset
        if self.el is not None and pylink is not None:
            try:
                self.el.sendMessage("FIX_ONSET")
            except Exception as e:
                print(f"Eyelink FIX_ONSET failed: {e}")
        core.wait(fixation_duration)
        
        # 2. 呈现静态Planko场景
        planks = self.draw_planks(trial_info['Planks'])
        
        # 绘制场景
        for plank in planks:
            plank.draw()
        
        # 绘制球的起始位置（每个trial自己的起点，可以在左或右）
        ball_start_x = trial_info.get('Ball_start_x', BALL_START_X)
        psychopy_ball_x = ball_start_x - SCREEN_WIDTH / 2
        psychopy_ball_y = SCREEN_HEIGHT / 2 - BALL_START_Y
        self.ball.pos = (psychopy_ball_x, psychopy_ball_y)
        self.ball.draw()
        
        # 绘制接球器（容器的底部和两侧壁）
        for stim in self.catcher_left_parts:
            stim.draw()
        for stim in self.catcher_right_parts:
            stim.draw()
        
        self.win.flip()
        stim_onset = core.getTime()
        trial_record['Stim_onset'] = stim_onset
        if self.el is not None and pylink is not None:
            try:
                self.el.sendMessage("STIM_ONSET")
            except Exception as e:
                print(f"Eyelink STIM_ONSET failed: {e}")
        
        # 3. 等待被试响应（F=左，J=右）
        self.kb.clock.reset()
        keys = event.waitKeys(keyList=['f', 'j', 'escape'])
        
        if 'escape' in keys:
            self.quit_experiment()
        
        response_time = core.getTime()
        trial_record['Response_time'] = response_time
        trial_record['RT'] = (response_time - stim_onset) * 1000  # 转换为ms
        trial_record['Choice'] = 'left' if 'f' in keys else 'right'
        trial_record['Correct'] = 1 if trial_record['Choice'] == trial_info['PhysOutcome'] else 0
        if self.el is not None and pylink is not None:
            try:
                self.el.sendMessage(
                    f"RESPONSE CHOICE {trial_record['Choice']} RT {int(trial_record['RT'])}"
                )
            except Exception as e:
                print(f"Eyelink RESPONSE marker failed: {e}")
        
        # 4. 信心度报告（数字键输入模式）
        confidence_start = core.getTime()
        confidence_input = ''  # 存储输入的数字
        confirmed = False
        
        while not confirmed:
            # 显示提示
            self.confidence_prompt.text = '请输入您的信心度（0-100）'
            self.confidence_prompt.draw()
            
            # 显示当前输入的值
            if confidence_input:
                self.confidence_value_text.text = f'{confidence_input}%'
            else:
                self.confidence_value_text.text = '_'
            self.confidence_value_text.draw()
            
            # 显示操作说明
            self.confidence_instruction.text = '输入数字（0-100），按空格键确认，按退格键删除'
            self.confidence_instruction.draw()
            
            self.win.flip()
            
            # 检查按键
            keys = event.getKeys()
            
            if 'escape' in keys:
                self.quit_experiment()
            elif 'backspace' in keys:
                # 删除最后一个数字
                confidence_input = confidence_input[:-1]
            elif 'space' in keys:
                # 按空格确认输入
                if confidence_input:
                    val = int(confidence_input)
                    if 0 <= val <= 100:
                        confirmed = True
            else:
                # 处理数字输入
                for key in keys:
                    if key.isdigit() and len(confidence_input) < 3:
                        # 限制最多3位数
                        temp = confidence_input + key
                        if int(temp) <= 100:
                            confidence_input = temp
        
        confidence_response_time = core.getTime()
        trial_record['Confidence_response_time'] = confidence_response_time
        trial_record['Confidence_RT'] = (confidence_response_time - response_time) * 1000
        trial_record['Confidence'] = int(confidence_input)
        if self.el is not None and pylink is not None:
            try:
                self.el.sendMessage("CONF_END")
            except Exception as e:
                print(f"Eyelink CONF_END failed: {e}")
        
        # 5. 反馈：播放真实下落动画
        feedback_onset = core.getTime()
        trial_record['Feedback_onset'] = feedback_onset
        if self.el is not None and pylink is not None:
            try:
                self.el.sendMessage("FEEDBACK_ONSET")
            except Exception as e:
                print(f"Eyelink FEEDBACK_ONSET failed: {e}")
        
        trajectory = json.loads(trial_info['True_trajectory'])
        ball_drop_start = core.getTime()
        trial_record['Ball_drop_start'] = ball_drop_start
        
        # 播放动画
        for point in trajectory:
            # 清屏并绘制场景
            for plank in planks:
                plank.draw()
            # 绘制接球器（容器的底部和两侧壁）
            for stim in self.catcher_left_parts:
                stim.draw()
            for stim in self.catcher_right_parts:
                stim.draw()
            
            # 更新球位置
            psychopy_x = point['x'] - SCREEN_WIDTH / 2
            psychopy_y = SCREEN_HEIGHT / 2 - point['y']
            self.ball.pos = (psychopy_x, psychopy_y)
            self.ball.draw()
            
            self.win.flip()
            core.wait(1.0 / FPS)  # 按照物理模拟的帧率播放
        
        ball_drop_end = core.getTime()
        trial_record['Ball_drop_end'] = ball_drop_end
        if self.el is not None and pylink is not None:
            try:
                self.el.sendMessage("FEEDBACK_END")
            except Exception as e:
                print(f"Eyelink FEEDBACK_END failed: {e}")
        
        # 5. 轨迹播放结束后直接进入ITI（不再单独文字反馈）
        
        # 6. ITI
        self.win.flip()  # 空屏
        core.wait(ITI_DURATION / 1000.0)
        # 结束本trial的Eyelink记录并写入结果标记
        if self.el is not None and pylink is not None:
            try:
                el = self.el
                choice = trial_record['Choice']
                correct = int(trial_record['Correct'])
                conf = int(trial_record['Confidence'])
                rt = int(trial_record['RT'])
                el.sendMessage(
                    f"TRIAL_RESULT CHOICE {choice} CORRECT {correct} "
                    f"CONF {conf} RT {rt}"
                )
                el.sendMessage("TRIAL_END")
                el.stopRecording()
                pylink.msecDelay(20)
            except Exception as e:
                print(f"Eyelink trial stop failed: {e}")

        return trial_record
    
    def run_block(self, block_id):
        """运行一个block"""
        block_trials = self.trial_sequence[
            self.trial_sequence['BlockID'] == block_id
        ]
        
        if self.el is not None and pylink is not None:
            try:
                self.el.sendMessage(f"BLOCK_START {block_id}")
            except Exception as e:
                print(f"Eyelink block start marker failed: {e}")
        
        # Block开始提示
        self.instruction_text.text = f"Block {block_id} / {N_BLOCKS}\n\n准备好后按空格键开始"
        self.instruction_text.color = 'white'
        self.instruction_text.draw()
        self.win.flip()
        event.waitKeys(keyList=['space'])
        
        # 运行block中的所有试次
        for _, trial_info in block_trials.iterrows():
            trial_record = self.run_trial(trial_info)
            self.trial_data.append(trial_record)
            
            # 实时保存数据（防止意外中断）
            self.save_data()
        
        # Block结束，休息
        if block_id < N_BLOCKS:
            self.instruction_text.text = (
                f"Block {block_id} 完成！\n\n"
                "请休息1-2分钟。\n"
                "实验人员将重新进行眼动校准。\n\n"
                "完成后按空格键继续。"
            )
            self.instruction_text.draw()
            self.win.flip()
            event.waitKeys(keyList=['space'])

        if self.el is not None and pylink is not None:
            try:
                self.el.sendMessage(f"BLOCK_END {block_id}")
            except Exception as e:
                print(f"Eyelink block end marker failed: {e}")
    
    def run(self):
        """运行整个实验"""
        try:
            self.setup_psychopy()
            self.setup_eyelink()
            self.show_instructions()
            
            # 运行所有blocks
            for block_id in range(1, N_BLOCKS + 1):
                self.run_block(block_id)
            
            # 实验结束
            self.instruction_text.text = "实验完成！\n\n感谢您的参与！"
            self.instruction_text.draw()
            self.win.flip()
            core.wait(3.0)
            
        finally:
            self.save_data()
            self.cleanup()
    
    def save_data(self):
        """保存行为数据"""
        if not self.trial_data:
            return
        
        df = pd.DataFrame(self.trial_data)
        output_file = self.output_dir / 'behavioral.csv'
        df.to_csv(output_file, index=False)
    
    def quit_experiment(self):
        """中途退出实验"""
        self.save_data()
        self.cleanup()
        core.quit()
    
    def cleanup(self):
        """清理资源"""
        if self.win:
            self.win.close()
        # 关闭Eyelink并接收EDF文件
        if self.el is not None and pylink is not None:
            try:
                try:
                    self.el.setOfflineMode()
                    pylink.msecDelay(50)
                except Exception:
                    pass

                if self.edf_file:
                    local_edf = self.output_dir / self.edf_file
                    try:
                        self.el.closeDataFile()
                    except Exception:
                        pass
                    try:
                        self.el.receiveDataFile(self.edf_file, str(local_edf))
                        print(f"EDF文件已保存到: {local_edf}")
                    except Exception as e:
                        print(f"接收EDF文件失败: {e}")

                self.el.close()
            except Exception as e:
                print(f"Eyelink 清理失败: {e}")
            finally:
                self.el = None


def main():
    """主函数"""
    # 1. 收集被试信息
    print("收集被试信息...")
    dialog = SubjectInfoDialog()
    subject_info = dialog.show()
    
    if subject_info is None:
        print("用户取消，退出程序")
        return
    
    print(f"被试信息: {subject_info}")
    
    # 2. 检查刺激文件是否存在（相对于脚本所在目录）
    primary_path = SCRIPT_DIR / 'stimuli' / 'stimulus_config.csv'
    alt_path = SCRIPT_DIR / 'stimulus_config.csv'

    if primary_path.exists():
        stimuli_file = primary_path
    elif alt_path.exists():
        stimuli_file = alt_path
    else:
        print(f"错误: 找不到刺激文件:\n  {primary_path}\n  或 {alt_path}")
        print("请先在同一目录下运行 stimulus_generator.py 生成刺激，"
              "并确认文件名为 stimulus_config.csv。")
        return
    
    # 3. 创建实验对象并运行
    print("初始化实验...")
    exp = PlankoExperiment(subject_info, str(stimuli_file))
    
    print("开始实验...")
    exp.run()
    
    print("实验结束")


if __name__ == '__main__':
    main()
