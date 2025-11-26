
"""
刺激文件一键修复工具

【自动处理所有问题】
✓ 重复的BoardID → 自动删除重复项
✓ 试次数不足 → 自动补充到目标数量
✓ BoardID不连续 → 自动重整为连续编号
✓ 试次数超标 → 自动随机删除多余项

【使用方法】
python "修复刺激文件.py"

一键解决所有问题，无需其他脚本！
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import shutil

from config import *
from stimulus_generator import generate_board_for_condition

def fix_stimulus_file():
    """一键修复刺激文件的所有问题"""
    csv_path = Path(__file__).parent / 'stimuli' / 'stimulus_config.csv'
    
    # 检查文件是否存在
    if not csv_path.exists():
        print("✗ 错误: 找不到 stimulus_config.csv")
        print("请先运行 stimulus_generator.py 生成初始刺激文件")
        return
    # 读取数据
    df = pd.read_csv(csv_path)
    
    print("=" * 60)
    print("刺激文件诊断与修复")
    print("=" * 60)
    
    print(f"\n【初始状态】")
    print(f"总试次数: {len(df)}")
    print(f"唯一BoardID数: {df['BoardID'].nunique()}")
    
    target = TRIALS_PER_CONDITION
    # ambiguous和congruent严格要求，conflict只要求总数
    strict_hv_types = ['congruent', 'ambiguous']
    loose_hv_type = 'conflict'
    
    total_target_strict = len(strict_hv_types) * SU_LEVELS * target  # 每个条件×每个SU level
    total_target_loose = SU_LEVELS * target  # conflict总数
    total_target = total_target_strict + total_target_loose
    
    # 问题1: 处理重复的BoardID
    if df['BoardID'].duplicated().any():
        dup_count = df['BoardID'].duplicated().sum()
        print(f"\n【问题1】发现 {dup_count} 个重复的BoardID")
        print("→ 解决方案: 删除重复项（保留第一个）")
        
        df = df.drop_duplicates(subset='BoardID', keep='first')
        print(f"✓ 已删除重复项，剩余 {len(df)} 个试次")
    
    # 当前分布
    counts = df.groupby(['HV_primary', 'SU_level']).size().unstack(fill_value=0)
    print(f"\n【当前分布】")
    print(counts)
    print(f"\n目标说明:")
    print(f"  - ambiguous/congruent: 每个SU level {target} 个，共 {total_target_strict} 个")
    print(f"  - conflict: 总共约 {total_target_loose} 个，SU分布不严格限制")
    print(f"  - 总计: {total_target} 个")
    
    # 问题2: 处理超标试次（只对strict类型检查每个SU level）
    excess_cells = []
    
    # 对ambiguous和congruent：检查每个SU level
    for hv in strict_hv_types:
        if hv in counts.index:
            for su in range(1, SU_LEVELS + 1):
                current = counts.loc[hv, su] if su in counts.columns else 0
                if current > target:
                    excess_cells.append((hv, su, current - target))
    
    # 对conflict：只检查总数
    if loose_hv_type in counts.index:
        conflict_total = counts.loc[loose_hv_type].sum()
        if conflict_total > total_target_loose:
            # 随机选择一些conflict试次删除
            excess_count = conflict_total - total_target_loose
            print(f"\n【问题2】conflict总数超标 {int(excess_count)} 个")
            mask = df['HV_primary'] == loose_hv_type
            indices = df[mask].index
            remove_indices = np.random.choice(indices, size=int(excess_count), replace=False)
            df = df.drop(remove_indices)
            print(f"  ✓ 已随机删除 {int(excess_count)} 个conflict试次")
    
    if excess_cells:
        print(f"\n【问题2】某些strict组合超标")
        for hv, su, count in excess_cells:
            print(f"  {hv}, SU_level={su}: 超出 {int(count)} 个")
            # 随机删除多余的试次
            mask = (df['HV_primary'] == hv) & (df['SU_level'] == su)
            indices = df[mask].index
            remove_indices = np.random.choice(indices, size=int(count), replace=False)
            df = df.drop(remove_indices)
            print(f"  ✓ 已随机删除 {int(count)} 个")
    
    counts = df.groupby(['HV_primary', 'SU_level']).size().unstack(fill_value=0)
    
    # 问题3: 补充缺失试次
    missing_cells = []
    
    # 对ambiguous和congruent：严格检查每个SU level
    for hv in strict_hv_types:
        for su in range(1, SU_LEVELS + 1):
            current = counts.loc[hv, su] if (hv in counts.index and su in counts.columns) else 0
            if current < target:
                missing_cells.append((hv, su, target - current))
    
    # 对conflict：只检查总数
    conflict_total = counts.loc[loose_hv_type].sum() if loose_hv_type in counts.index else 0
    if conflict_total < total_target_loose:
        conflict_missing = total_target_loose - conflict_total
        print(f"\n【问题3】conflict总数不足，需补充约 {int(conflict_missing)} 个")
    
    if missing_cells:
        total_missing_strict = sum(count for _, _, count in missing_cells)
        print(f"\n【问题3】strict组合试次不足，需补充 {int(total_missing_strict)} 个")

        su_bins = np.linspace(0, 1, SU_LEVELS + 1)

        # 为每个(HV, SU_level)组合单独控制总尝试次数，避免无休止搜索
        MAX_ATTEMPTS_PER_CELL = 400
        ATTEMPTS_PER_CALL = 100

        new_boards = []

        for hv, su_level, count in missing_cells:
            su_min = su_bins[su_level - 1]
            su_max = su_bins[su_level]
            need = int(count)
            attempts_left = MAX_ATTEMPTS_PER_CELL

            print(f"  {hv}, SU_level={su_level}: 需要 {need} 个 (本轮最多尝试 {attempts_left} 次)")

            generated_here = 0
            attempts_used = 0
            while need > 0 and attempts_left > 0:
                this_attempts = min(ATTEMPTS_PER_CALL, attempts_left)
                board = generate_board_for_condition(hv, su_min, su_max, max_attempts=this_attempts)
                attempts_left -= this_attempts
                attempts_used += this_attempts

                # 适度的进度输出，避免长时间无反应
                if attempts_used % (ATTEMPTS_PER_CALL * 2) == 0 or attempts_left == 0:
                    print(f"    已尝试 {attempts_used}/{MAX_ATTEMPTS_PER_CELL} 次，当前成功 {generated_here} 个")

                if board is None:
                    continue

                board['SU_level'] = su_level
                new_boards.append(board)
                generated_here += 1
                need -= 1

            print(f"    -> 本轮为 {hv}, SU_level={su_level} 成功生成 {generated_here} 个，剩余 {need} 个未补上")
        
        # 如果conflict不足，随机生成一些（SU level随机分布）
        if loose_hv_type in counts.index:
            conflict_total = counts.loc[loose_hv_type].sum()
        else:
            conflict_total = 0
        
        if conflict_total < total_target_loose:
            conflict_need = int(total_target_loose - conflict_total)
            print(f"\n  补充conflict: 需要 {conflict_need} 个（SU随机分布）")
            
            conflict_generated = 0
            attempts = 0
            max_attempts = conflict_need * 20
            
            while conflict_generated < conflict_need and attempts < max_attempts:
                # 随机选择SU level
                su_level = np.random.randint(1, SU_LEVELS + 1)
                su_min = su_bins[su_level - 1]
                su_max = su_bins[su_level]
                
                board = generate_board_for_condition(loose_hv_type, su_min, su_max, max_attempts=50)
                attempts += 50
                
                if board is not None:
                    board['SU_level'] = su_level
                    new_boards.append(board)
                    conflict_generated += 1
                
                if attempts % 200 == 0:
                    print(f"    已尝试 {attempts} 次，成功 {conflict_generated} 个")
            
            print(f"    -> conflict成功生成 {conflict_generated} 个")

        # 把所有成功生成的新试次立刻并入df
        if new_boards:
            new_records = []
            for board in new_boards:
                record = {
                    'BoardID': 'TEMP',
                    'SU_raw': board['su_raw'],
                    'SU_level': board['SU_level'],
                    'HV_primary': board['hv_primary'],
                    'HV_secondary_count': board['hv_secondary_count'],
                    'HV_secondary_collision': board['hv_secondary_collision'],
                    'PhysOutcome': board['phys_outcome'],
                    'Ball_start_x': board['ball_start_x'],
                    'Ball_start_y': BALL_START_Y,
                    'Ball_start_side': board['ball_start_side'],
                    'True_trajectory': json.dumps(board['trajectory']),
                    'Planks': json.dumps(board['planks'])
                }
                new_records.append(record)

            new_df = pd.DataFrame(new_records)
            df = pd.concat([df, new_df], ignore_index=True)
            print(f"✓ 本轮共补充 {len(new_boards)} 个试次")

    # 问题4: 重整BoardID为连续编号
    print(f"\n【最后一步】重整BoardID")
    df = df.sort_values(['HV_primary', 'SU_level']).reset_index(drop=True)
    df['BoardID'] = [f'B{i:04d}' for i in range(len(df))]
    print(f"✓ 已重整为 B0000 - B{len(df)-1:04d}")
    
    # 保存
    df.to_csv(csv_path, index=False)
    
    # 最终验证
    print(f"\n" + "=" * 60)
    print("【最终结果】")
    print("=" * 60)
    
    final_counts = df.groupby(['HV_primary', 'SU_level']).size().unstack(fill_value=0)
    print(final_counts)
    
    print(f"\n总试次数: {len(df)} / {total_target}")
    print(f"唯一BoardID: {df['BoardID'].nunique()}")
    print(f"BoardID范围: {df['BoardID'].iloc[0]} - {df['BoardID'].iloc[-1]}")
    print(f"是否有重复: {'是' if df['BoardID'].duplicated().any() else '否'}")
    
    # 验证结果
    strict_ok = True
    for hv in strict_hv_types:
        if hv in final_counts.index:
            for su in range(1, SU_LEVELS + 1):
                if su in final_counts.columns:
                    if final_counts.loc[hv, su] != target:
                        strict_ok = False
                        break
    
    conflict_total = final_counts.loc[loose_hv_type].sum() if loose_hv_type in final_counts.index else 0
    conflict_ok = abs(conflict_total - total_target_loose) <= 5  # 允许±5的误差
    
    if strict_ok and conflict_ok and not df['BoardID'].duplicated().any():
        print(f"\n✓✓✓ 完美！文件已修复！")
        print(f"✓ ambiguous/congruent每个SU level都是 {target} 个试次")
        print(f"✓ conflict总数约 {conflict_total} 个（目标{total_target_loose}）")
        print(f"✓ BoardID连续且无重复")
    else:
        print(f"\n⚠ 仍存在问题，建议再次运行此脚本")
        if not strict_ok:
            print(f"  - ambiguous/congruent的SU分布不完全符合要求")
        if not conflict_ok:
            print(f"  - conflict总数 {conflict_total} 与目标 {total_target_loose} 差距较大")

if __name__ == '__main__':
    print("\n即将开始自动诊断和修复...")
    print("按Enter键开始，或Ctrl+C取消...")
    input()
    fix_stimulus_file()
