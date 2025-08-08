#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################

"""
Author: Tencent AI Arena Authors

"""

import numpy as np
import math
from agent_ppo.feature.definition import RelativeDistance, RelativeDirection, DirectionAngles, reward_process


def norm(v, max_v, min_v=0):
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)


# class Preprocessor:
#     def __init__(self) -> None:
#         self.move_action_num = 8
#         self.reset()

#     def reset(self):
#         self.step_no = 0
#         self.cur_pos = (0, 0)
#         self.cur_pos_norm = np.array((0, 0))
#         self.end_pos = None
#         self.is_end_pos_found = False
#         self.history_pos = []
#         self.bad_move_ids = set()

#     def _get_pos_feature(self, found, cur_pos, target_pos):
#         relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))
#         dist = np.linalg.norm(relative_pos)
#         target_pos_norm = norm(target_pos, 128, -128)
#         feature = np.array(
#             (
#                 found,
#                 norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
#                 norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
#                 target_pos_norm[0],
#                 target_pos_norm[1],
#                 norm(dist, 1.41 * 128),
#             ),
#         )
#         return feature

#     def pb2struct(self, frame_state, last_action):
#         obs, _ = frame_state
#         self.step_no = obs["frame_state"]["step_no"]

#         hero = obs["frame_state"]["heroes"][0]
#         self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])

#         # History position
#         # 历史位置
#         self.history_pos.append(self.cur_pos)
#         if len(self.history_pos) > 10:
#             self.history_pos.pop(0)

#         # End position
#         # 终点位置
#         for organ in obs["frame_state"]["organs"]:
#             if organ["sub_type"] == 4:
#                 end_pos_dis = RelativeDistance[organ["relative_pos"]["l2_distance"]]
#                 end_pos_dir = RelativeDirection[organ["relative_pos"]["direction"]]
#                 if organ["status"] != -1:
#                     self.end_pos = (organ["pos"]["x"], organ["pos"]["z"])
#                     self.is_end_pos_found = True
#                 # if end_pos is not found, use relative position to predict end_pos
#                 # 如果终点位置未找到，使用相对位置预测终点位置
#                 elif (not self.is_end_pos_found) and (
#                     self.end_pos is None
#                     or self.step_no % 100 == 0
#                     or self.end_pos_dir != end_pos_dir
#                     or self.end_pos_dis != end_pos_dis
#                 ):
#                     distance = end_pos_dis * 20
#                     theta = DirectionAngles[end_pos_dir]
#                     delta_x = distance * math.cos(math.radians(theta))
#                     delta_z = distance * math.sin(math.radians(theta))

#                     self.end_pos = (
#                         max(0, min(128, round(self.cur_pos[0] + delta_x))),
#                         max(0, min(128, round(self.cur_pos[1] + delta_z))),
#                     )

#                     self.end_pos_dir = end_pos_dir
#                     self.end_pos_dis = end_pos_dis

#         self.last_pos_norm = self.cur_pos_norm
#         self.cur_pos_norm = norm(self.cur_pos, 128, -128)
#         self.feature_end_pos = self._get_pos_feature(self.is_end_pos_found, self.cur_pos, self.end_pos)

#         # History position feature
#         # 历史位置特征
#         self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.history_pos[0])

#         self.move_usable = True
#         self.last_action = last_action

#     def process(self, frame_state, last_action):
#         self.pb2struct(frame_state, last_action)

#         # Legal action
#         # 合法动作
#         legal_action = self.get_legal_action()

#         # Feature
#         # 特征
#         feature = np.concatenate([self.cur_pos_norm, self.feature_end_pos, self.feature_history_pos, legal_action])

#         return (
#             feature,
#             legal_action,
#             reward_process(self.feature_end_pos[-1], self.feature_history_pos[-1]),
#         )

#     def get_legal_action(self):
#         # if last_action is move and current position is the same as last position, add this action to bad_move_ids
#         # 如果上一步的动作是移动，且当前位置与上一步位置相同，则将该动作加入到bad_move_ids中
#         if (
#             abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 0.001
#             and abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 0.001
#             and self.last_action > -1
#         ):
#             self.bad_move_ids.add(self.last_action)
#         else:
#             self.bad_move_ids = set()

#         legal_action = [self.move_usable] * self.move_action_num
#         for move_id in self.bad_move_ids:
#             legal_action[move_id] = 0

#         if self.move_usable not in legal_action:
#             self.bad_move_ids = set()
#             return [self.move_usable] * self.move_action_num

#         return legal_action

class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = 8  # 8个移动方向
        self.flash_action_id = 8  # 闪现动作ID=8
        self.total_action_num = 9  # 总动作数=9 (8移动+1闪现)
        self.reset()

    def reset(self):
        self.step_no = 0
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array((0, 0))
        self.end_pos = None
        self.is_end_pos_found = False
        self.history_pos = []
        self.bad_move_ids = set()
        
        # 闪现相关状态
        self.last_move_dir = None  # 记录上一次有效移动方向
        self.can_flash = False     # 闪现可用状态
        self.last_flash_step = -999 # 上次使用闪现的step

    def _get_pos_feature(self, found, cur_pos, target_pos):
        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))
        dist = np.linalg.norm(relative_pos)
        target_pos_norm = norm(target_pos, 128, -128)
        feature = np.array(
            (
                found,
                norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
                norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
                target_pos_norm[0],
                target_pos_norm[1],
                norm(dist, 1.41 * 128),
            ),
        )
        return feature

    def _get_nearest_treasure(self, organs):
        """计算到最近宝箱的归一化距离[0-1]"""
        min_dist = float('inf')
        for organ in organs:
            if organ['sub_type'] == 1:  # 宝箱类型
                dist = math.sqrt(
                    (organ['pos']['x'] - self.cur_pos[0])**2 +
                    (organ['pos']['z'] - self.cur_pos[1])**2
                )
                if dist < min_dist:
                    min_dist = dist
        return norm(min_dist, 20) if min_dist != float('inf') else 1.0

    def pb2struct(self, frame_state, last_action):
        obs, _ = frame_state
        self.step_no = obs["frame_state"]["step_no"]

        # 解析英雄状态
        hero = obs["frame_state"]["heroes"][0]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])

        # 解析闪现技能状态
        if 'talent' in hero:
            talent = hero['talent']
            self.can_flash = (talent['status'] == 1)  # 1表示可用
        
        # 记录有效移动方向（忽略闪现动作）
        if last_action is not None and last_action < self.move_action_num:
            self.last_move_dir = last_action

        # 历史位置记录
        self.history_pos.append(self.cur_pos)
        if len(self.history_pos) > 10:
            self.history_pos.pop(0)

        # 终点位置处理
        for organ in obs["frame_state"]["organs"]:
            if organ["sub_type"] == 4:  # 终点类型
                end_pos_dis = RelativeDistance[organ["relative_pos"]["l2_distance"]]
                end_pos_dir = RelativeDirection[organ["relative_pos"]["direction"]]
                if organ["status"] != -1:
                    self.end_pos = (organ["pos"]["x"], organ["pos"]["z"])
                    self.is_end_pos_found = True
                elif (not self.is_end_pos_found) and (
                    self.end_pos is None
                    or self.step_no % 100 == 0
                    or self.end_pos_dir != end_pos_dir
                    or self.end_pos_dis != end_pos_dis
                ):
                    distance = end_pos_dis * 20
                    theta = DirectionAngles[end_pos_dir]
                    delta_x = distance * math.cos(math.radians(theta))
                    delta_z = distance * math.sin(math.radians(theta))
                    self.end_pos = (
                        max(0, min(128, round(self.cur_pos[0] + delta_x))),
                        max(0, min(128, round(self.cur_pos[1] + delta_z))),
                    )
                    self.end_pos_dir = end_pos_dir
                    self.end_pos_dis = end_pos_dis

        # 位置特征计算
        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = norm(self.cur_pos, 128, -128)
        self.feature_end_pos = self._get_pos_feature(
            self.is_end_pos_found, self.cur_pos, self.end_pos)
        self.feature_history_pos = self._get_pos_feature(
            1, self.cur_pos, self.history_pos[0] if self.history_pos else self.cur_pos)

        self.move_usable = True
        self.last_action = last_action

    def get_legal_action(self):
        legal_actions = [1] * self.total_action_num  # 初始化所有动作
        
        # 移动动作合法性检查
        if (abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 0.001 and
            abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 0.001 and
            self.last_action is not None and 
            self.last_action < self.move_action_num):
            legal_actions[self.last_action] = 0  # 无效移动动作
        
        # 闪现动作合法性条件
        flash_legal = (
            self.can_flash and                    # 技能可用
            self.last_move_dir is not None and    # 有历史移动方向
            (self.step_no - self.last_flash_step) > 5  # 冷却限制
        )
        legal_actions[self.flash_action_id] = 1 if flash_legal else 0
        
        # 确保至少有一个合法动作
        if sum(legal_actions) == 0:
            legal_actions = [1] * self.total_action_num
            self.bad_move_ids = set()
        
        return legal_actions

    def process(self, frame_state, last_action):
        self.pb2struct(frame_state, last_action)
        
        # 获取最近宝箱距离
        obs, _ = frame_state
        nearest_treasure = self._get_nearest_treasure(obs["frame_state"]["organs"])
        
        # 生成合法动作掩码
        legal_action = self.get_legal_action()
        
        # 组合特征向量
        feature = np.concatenate([
            self.cur_pos_norm,          # 2
            self.feature_end_pos,       # 6
            self.feature_history_pos,   # 6
            np.array([float(self.can_flash)]),  # 转换为1维数组
            [nearest_treasure],         # 1 (宝箱距离)
            legal_action                # 9
        ])
        
        # 判断是否使用闪现
        used_flash = (last_action == self.flash_action_id)
        if used_flash:
            self.last_flash_step = self.step_no
        
        return (
            feature,  # 25维 (2+6+6+2+1+8=25)
            legal_action,
            reward_process(
                end_dist=self.feature_end_pos[-1],
                history_dist=self.feature_history_pos[-1],
                nearest_treasure_dist=nearest_treasure,
                used_flash=used_flash
            )
        )