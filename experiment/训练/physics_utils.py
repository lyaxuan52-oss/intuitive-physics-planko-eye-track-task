
import pymunk
import numpy as np
from config import *


def create_space():
    """创建Pymunk物理空间"""
    space = pymunk.Space()
    space.gravity = (0, GRAVITY)
    # 提高求解精度，减轻重叠/空气墙感
    space.iterations = 30
    # 轻微全局阻尼，抑制长时间来回乱弹
    space.damping = 0.99
    return space


def create_ball(space, x, y):
    """创建小球"""
    mass = BALL_MASS
    moment = pymunk.moment_for_circle(mass, 0, BALL_RADIUS)
    body = pymunk.Body(mass, moment)
    body.position = x, y
    shape = pymunk.Circle(body, BALL_RADIUS)
    shape.elasticity = ELASTICITY
    shape.friction = FRICTION
    # 为后续碰撞检测设置类型
    shape.collision_type = 10
    space.add(body, shape)
    return body, shape


def create_plank(space, x, y, angle, width=PLANK_WIDTH, height=PLANK_HEIGHT):
    """创建静态挡板"""
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = x, y
    body.angle = np.radians(angle)
    
    # 创建矩形挡板
    vertices = [
        (-width/2, -height/2),
        (width/2, -height/2),
        (width/2, height/2),
        (-width/2, height/2)
    ]
    shape = pymunk.Poly(body, vertices)
    shape.elasticity = ELASTICITY
    shape.friction = FRICTION
    space.add(body, shape)
    return body, shape


def create_catchers(space):
    """创建两个接球器（静态传感器）"""
    catchers = {}
    
    wall_height = 80
    wall_width = 10

    # 左侧接球器底部
    left_bottom_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    left_bottom_body.position = CATCHER_LEFT_X, CATCHER_Y
    left_bottom_shape = pymunk.Poly.create_box(left_bottom_body, (CATCHER_WIDTH, CATCHER_HEIGHT))
    left_bottom_shape.elasticity = ELASTICITY
    left_bottom_shape.friction = FRICTION
    space.add(left_bottom_body, left_bottom_shape)

    wall_center_y = CATCHER_Y - wall_height / 2.0

    # 左侧接球器侧壁
    left_wall_left_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    left_wall_left_body.position = CATCHER_LEFT_X - CATCHER_WIDTH / 2.0 + wall_width / 2.0, wall_center_y
    left_wall_left_shape = pymunk.Poly.create_box(left_wall_left_body, (wall_width, wall_height))
    left_wall_left_shape.elasticity = ELASTICITY
    left_wall_left_shape.friction = FRICTION
    space.add(left_wall_left_body, left_wall_left_shape)

    left_wall_right_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    left_wall_right_body.position = CATCHER_LEFT_X + CATCHER_WIDTH / 2.0 - wall_width / 2.0, wall_center_y
    left_wall_right_shape = pymunk.Poly.create_box(left_wall_right_body, (wall_width, wall_height))
    left_wall_right_shape.elasticity = ELASTICITY
    left_wall_right_shape.friction = FRICTION
    space.add(left_wall_right_body, left_wall_right_shape)

    catchers['left'] = (left_bottom_body, left_bottom_shape)
    
    # 右侧接球器底部
    right_bottom_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    right_bottom_body.position = CATCHER_RIGHT_X, CATCHER_Y
    right_bottom_shape = pymunk.Poly.create_box(right_bottom_body, (CATCHER_WIDTH, CATCHER_HEIGHT))
    right_bottom_shape.elasticity = ELASTICITY
    right_bottom_shape.friction = FRICTION
    space.add(right_bottom_body, right_bottom_shape)

    # 右侧接球器侧壁
    right_wall_left_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    right_wall_left_body.position = CATCHER_RIGHT_X - CATCHER_WIDTH / 2.0 + wall_width / 2.0, wall_center_y
    right_wall_left_shape = pymunk.Poly.create_box(right_wall_left_body, (wall_width, wall_height))
    right_wall_left_shape.elasticity = ELASTICITY
    right_wall_left_shape.friction = FRICTION
    space.add(right_wall_left_body, right_wall_left_shape)

    right_wall_right_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    right_wall_right_body.position = CATCHER_RIGHT_X + CATCHER_WIDTH / 2.0 - wall_width / 2.0, wall_center_y
    right_wall_right_shape = pymunk.Poly.create_box(right_wall_right_body, (wall_width, wall_height))
    right_wall_right_shape.elasticity = ELASTICITY
    right_wall_right_shape.friction = FRICTION
    space.add(right_wall_right_body, right_wall_right_shape)

    catchers['right'] = (right_bottom_body, right_bottom_shape)
    
    return catchers


def run_simulation(space, ball_body, ball_shape, planks=None, max_time=10.0):
    """运行物理模拟并返回球的落点、轨迹和第一次碰撞位置

    返回: (landing, trajectory, first_collision_side)
        landing: 'left', 'right', or 'none'
        trajectory: 轨迹点列表
        first_collision_side: 第一次碰撞的挡板在哪侧 ('left', 'right', or 'none')
    """

    # 使用较小的子步长提高碰撞精度，减少穿板/空气墙
    visual_dt = 1.0 / FPS
    substeps = 20
    dt = visual_dt / substeps

    t = 0.0
    trajectory = []
    landing = 'none'
    first_collision_side = 'none'
    first_collision_detected = False

    prev_vy = float(ball_body.velocity.y)

    # 预先计算接球器底板的几何边界（只用顶部边界做“打到底”的判定）
    half_w = CATCHER_WIDTH / 2.0
    half_h = CATCHER_HEIGHT / 2.0
    left_top_y = CATCHER_Y - half_h
    right_top_y = CATCHER_Y - half_h  # 两个杯底高度相同

    while t < max_time and landing == 'none':
        # 在子步中精细检测是否跨越接球器底板的“上边界”
        prev_bottom_y = float(ball_body.position.y + BALL_RADIUS)

        for _ in range(substeps):
            space.step(dt)

            x_sub, y_sub = ball_body.position
            curr_bottom_y = float(y_sub + BALL_RADIUS)

            if landing == 'none':
                # 左杯：球底边从上方跨越到杯底上边界以下，且水平投影落在杯底宽度范围内
                if (
                    (prev_bottom_y < left_top_y <= curr_bottom_y)
                    and (CATCHER_LEFT_X - half_w <= x_sub <= CATCHER_LEFT_X + half_w)
                ):
                    landing = 'left'

                # 右杯：同理
                elif (
                    (prev_bottom_y < right_top_y <= curr_bottom_y)
                    and (CATCHER_RIGHT_X - half_w <= x_sub <= CATCHER_RIGHT_X + half_w)
                ):
                    landing = 'right'

            if landing != 'none':
                # 一旦击中接球器底部上边界，立刻停球，避免在杯内乱弹或飞到另一侧
                ball_body.velocity = (0.0, 0.0)
                break

        x, y = ball_body.position
        vx, vy = ball_body.velocity

        # 检测第一次碰撞（通过速度突变判断），用于次启发式
        if not first_collision_detected:
            if abs(vy - prev_vy) > 50:
                if planks:
                    screen_center = SCREEN_WIDTH / 2
                    if x < screen_center:
                        first_collision_side = 'left'
                    else:
                        first_collision_side = 'right'
                    first_collision_detected = True
        prev_vy = vy

        trajectory.append({
            'x': float(x),
            'y': float(y),
            't': float(t),
            'vx': float(vx),
            'vy': float(vy),
        })

        # 已经落入接球器底部，结束模拟
        if landing != 'none':
            break

        # 掉出屏幕或时间过长则停止
        if y > SCREEN_HEIGHT + 200:
            break

        # 球几乎静止且已经低于接球器区域，也停止
        if abs(vx) < 1 and abs(vy) < 1 and y > CATCHER_Y + CATCHER_HEIGHT:
            break

        t += visual_dt

    return landing, trajectory, first_collision_side


def generate_random_planks(n_planks=N_PLANKS, seed=None):

    if seed is not None:
        np.random.seed(seed)
    
    planks = []
    for _ in range(n_planks):
        x = np.random.uniform(SCREEN_WIDTH * 0.2, SCREEN_WIDTH * 0.8)
        y = np.random.uniform(PLANK_Y_MIN, PLANK_Y_MAX)
        angle = np.random.uniform(-45, 45)  # degrees
        planks.append({'x': x, 'y': y, 'angle': angle})
    
    return planks


def check_straight_line_heuristic(ball_x, ball_y, catcher_x, catcher_y, planks):
    """
    检查是否存在从球到接球器的直线路径（无碰撞）
    简化版本：检查直线段是否与任何挡板相交
    返回: True if 直线路径存在（启发式有效）
    """
    # 这里使用简化的几何检测
    # 实际实现可以更精确
    for plank in planks:
        px, py = plank['x'], plank['y']
        angle = np.radians(plank['angle'])
        
        # 计算挡板的端点
        half_w = PLANK_WIDTH / 2
        dx = half_w * np.cos(angle)
        dy = half_w * np.sin(angle)
        
        p1 = (px - dx, py - dy)
        p2 = (px + dx, py + dy)
        
        # 检查直线段 (ball_x, ball_y) -> (catcher_x, catcher_y) 是否与挡板相交
        if line_segment_intersect((ball_x, ball_y), (catcher_x, catcher_y), p1, p2):
            return False  # 有碰撞，启发式无效
    
    return True  # 无碰撞，启发式有效


def line_segment_intersect(A, B, C, D):
    """
    检查线段AB和线段CD是否相交
    """
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def count_planks_per_side(planks, center_x=SCREEN_WIDTH//2):
    """
    统计左右两侧的挡板数量
    """
    left_count = sum(1 for p in planks if p['x'] < center_x)
    right_count = sum(1 for p in planks if p['x'] >= center_x)
    return left_count, right_count


def calculate_plank_total_length_per_side(planks, center_x=SCREEN_WIDTH//2):
    """
    计算左右两侧挡板的总长度
    返回: (left_total_length, right_total_length)
    
    次启发式逻辑：哪边挡板总长度更长，球更可能被挡到对面
    """
    left_total = sum(PLANK_WIDTH for p in planks if p['x'] < center_x)
    right_total = sum(PLANK_WIDTH for p in planks if p['x'] >= center_x)
    return left_total, right_total


def apply_jitter_to_planks(planks):
    """对挡板配置应用位置和角度抖动"""
    jittered = []
    for plank in planks:
        jittered.append({
            'x': plank['x'] + np.random.normal(0, JITTER_POSITION_STD),
            'y': plank['y'] + np.random.normal(0, JITTER_POSITION_STD),
            'angle': plank['angle'] + np.random.normal(0, JITTER_ANGLE_STD)
        })
    return jittered


def circle_intersects_rect(cx, cy, radius, rect_cx, rect_cy, rect_w, rect_h):
    half_w = rect_w / 2.0
    half_h = rect_h / 2.0
    closest_x = min(max(cx, rect_cx - half_w), rect_cx + half_w)
    closest_y = min(max(cy, rect_cy - half_h), rect_cy + half_h)
    dx = cx - closest_x
    dy = cy - closest_y
    return dx * dx + dy * dy <= radius * radius
