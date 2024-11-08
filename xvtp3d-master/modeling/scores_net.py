import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import json

from modeling.lib import PointSubGraph, CrossAttention, MLP

from utils.func_lab import trans_3D_2D, get_neighbour_3Dpoints, get_dis
from utils.calibration import Calibration


class DecoderRes(nn.Module):  # 全连接+残差
    def __init__(self, hidden_size, out_features=60):
        super(DecoderRes, self).__init__()
        self.mlp = MLP(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.mlp(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class DecoderResCat(nn.Module):  # 全连接+残差
    def __init__(self, hidden_size, in_features, out_features=60):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class ScoresNet(nn.Module):
    '''
        这个类是用来计算得分的，包括两个阶段的得分计算，第一个阶段是车道线的得分，第二个阶段是3D目标的得分
    '''
    def __init__(self, cfg, device, calib):
        super(ScoresNet, self).__init__()
        self.cfg = cfg
        hidden_size = cfg.MODEL.hidden_size
        self.future_frame_num = cfg.DATA.pred_len
        self.mode_num = cfg.MODEL.mode_num
        self.device = device
        self.hidden_size = hidden_size
        self.m = cfg.modality

        self.calib = calib

        if self.cfg.MODEL.share_weight:
            self.goals_2D_cross_attention = CrossAttention(hidden_size)
            self.goals_2D_point_sub_graph = PointSubGraph(hidden_size)
            self.stage_one_cross_attention = CrossAttention(hidden_size)
            self.stage_one_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=1)
            self.stage_one_goals_2D_decoder = DecoderResCat(hidden_size, hidden_size * 4, out_features=hidden_size)
        else:
            self.goals_2D_cross_attention_b = CrossAttention(hidden_size)
            self.goals_2D_cross_attention_f = CrossAttention(hidden_size)
            self.goals_2D_point_sub_graph_b = PointSubGraph(hidden_size)
            self.goals_2D_point_sub_graph_f = PointSubGraph(hidden_size)
            self.stage_one_cross_attention_b = CrossAttention(hidden_size)
            self.stage_one_cross_attention_f = CrossAttention(hidden_size)
            self.stage_one_decoder_b = DecoderResCat(hidden_size, hidden_size * 3, out_features=1)
            self.stage_one_decoder_f = DecoderResCat(hidden_size, hidden_size * 3, out_features=1)
            self.stage_one_goals_2D_decoder_b = DecoderResCat(hidden_size, hidden_size * 4, out_features=hidden_size)
            self.stage_one_goals_2D_decoder_f = DecoderResCat(hidden_size, hidden_size * 4, out_features=hidden_size)

        if cfg.MODEL.self_rein:
            self.reinforce_attention_b = CrossAttention(hidden_size)
            self.reinforce_attention_f = CrossAttention(hidden_size)

        if cfg.MODEL.cross_enc:
            self.b2f_cross_attn = CrossAttention(hidden_size)
            self.f2b_cross_attn = CrossAttention(hidden_size)

        # get score map across view
        if cfg.MODEL.cross_type == 0:
            self.score_mlp = DecoderResCat(hidden_size, hidden_size, out_features=1)  # 这样不能用MLP因为MLP自带LN，会使结果全0
        elif cfg.MODEL.cross_type == 1:
            self.score_mlp = DecoderResCat(hidden_size, hidden_size, out_features=1)
        elif cfg.MODEL.cross_type == 2:
            self.score_mlp = DecoderResCat(hidden_size, hidden_size * 2, out_features=1)
        elif cfg.MODEL.cross_type == 3:
            self.score_mlp_b = DecoderResCat(hidden_size, hidden_size, out_features=2)
            self.score_mlp_f = DecoderResCat(hidden_size, hidden_size, out_features=2)
            self.norm_act = nn.Sigmoid()  # todo: 换成tanh因为sigmoid可能与exp不太搭调

    def lane_score_net(self, i, lane_states_batch, inputs, inputs_lengths, hidden_state, lane_input_mask, tag):
        '''
            这个函数是用来计算车道线的得分的
            i: 第i个样本
            lane_states_batch: 批量的车道线特征
            inputs: 输入的特征
            inputs_lengths: 输入的长度
            hidden_state: 隐藏状态
            lane_input_mask: 车道线的mask
            tag: 标签
        '''
        if self.cfg.MODEL.share_weight:# 如果共享权重
            cross_attn = self.stage_one_cross_attention # 第一阶段的交叉注意力
            stage_decoder = self.stage_one_decoder # 第一阶段的解码器
        else:# 如果不共享权重
            if tag == 'bev':# 如果是bev
                cross_attn = self.stage_one_cross_attention_b # 第一阶段的bev交叉注意力
                stage_decoder = self.stage_one_decoder_b # 第一阶段的bev解码器
            else:
                cross_attn = self.stage_one_cross_attention_f # 第一阶段的fpv交叉注意力
                stage_decoder = self.stage_one_decoder_f # 第一阶段的fpv解码器

        stage_one_hidden = lane_states_batch[i] # [n_lane,128]
        stage_one_hidden_attention = cross_attn(
            stage_one_hidden.unsqueeze(0), inputs[i][:inputs_lengths[i]].unsqueeze(0)).squeeze(0)  # [n_lane,128], [n_lane+n_ped,128] -> [n_lane,128]，线性编码
        stage_one_scores = stage_decoder(torch.cat([hidden_state[0, :].unsqueeze(0).expand(
            stage_one_hidden.shape), stage_one_hidden, stage_one_hidden_attention], dim=-1))  # [n_lane,1]，解码获取得分
        stage_one_scores = stage_one_scores.squeeze(-1) # [n_lane]
        mask = torch.tensor(lane_input_mask[i], dtype=torch.bool, device=self.device) # [n_lane]，mask的作用是将不需要的车道线的得分设置为负无穷
        stage_one_scores.masked_fill(mask, -np.inf)
        stage_one_scores = F.log_softmax(stage_one_scores, dim=-1)# log_softmax和softmax的区别是log_softmax是对softmax的结果取对数
        return stage_one_scores  # [n_lane]

    def lane_scoring(self, i, mapping_b, mapping_f, loss_b, loss_f, f_lane_b, f_lane_f, l_masks_b, l_masks_f,
                     inputs_b, inputs_f, lens_b, lens_f, h_state_b, h_state_f):
        '''
            这个函数是用来计算车道线的得分的
            i: 第i个样本
            mapping_b: bev的映射
            mapping_f: fpv的映射
            loss_b: bev的loss
            loss_f: fpv的loss
            f_lane_b: bev的车道线特征
            f_lane_f: fpv的车道线特征
            l_masks_b: bev的mask
            l_masks_f: fpv的mask
            inputs_b: bev的输入
            inputs_f: fpv的输入
            lens_b: bev的长度
            lens_f: fpv的长度
            h_state_b: bev的隐藏状态
            h_state_f: fpv的隐藏状态
        '''
        if (self.m == 'both') or (self.m == 'bev'):
            lane_score_b = self.lane_score_net(i, f_lane_b, inputs_b, lens_b, h_state_b, l_masks_b, 'bev') # [n_lane]
            loss_b[i] += F.nll_loss(lane_score_b.unsqueeze(0),
                                    torch.tensor([mapping_b[i]['stage_one_label']], device=self.device))# 将车道线的得分和真实的车道线的标签进行比较
            n_lane = lane_score_b.shape[0]# 车道线的数量
            _, topk_ids_b = torch.topk(lane_score_b, k=n_lane)# 获取车道线得分最高的车道线
        else:
            topk_ids_b = None

        if (self.m == 'both') or (self.m == 'fpv'):
            lane_score_f = self.lane_score_net(i, f_lane_f, inputs_f, lens_f, h_state_f, l_masks_f, 'fpv')# [n_lane]
            loss_f[i] += F.nll_loss(lane_score_f.unsqueeze(0),
                                    torch.tensor([mapping_f[i]['stage_one_label']], device=self.device))
            n_lane = lane_score_f.shape[0]
            _, topk_ids_f = torch.topk(lane_score_f, k=n_lane)
        else:
            topk_ids_f = None

        # Select top K lanes, where K is dynamic. The sum of the probabilities of selected lanes is larger than threshold (0.95).
        # 从得分最高的车道线开始累加，一直往得分较低的车道线累加，直到累加和超过设定的阈值（0.95）

        threshold = 0.95
        if topk_ids_b is not None:
            sum = 0.0
            for idx, each in enumerate(torch.exp(lane_score_b[topk_ids_b])):
                sum += each
                if sum > threshold:
                    topk_ids_b = topk_ids_b[:idx + 1]
                    break
        if topk_ids_f is not None:
            sum = 0.0
            for idx, each in enumerate(torch.exp(lane_score_f[topk_ids_f])):
                sum += each
                if sum > threshold:
                    topk_ids_f = topk_ids_f[:idx + 1]
                    break

        if self.m == 'bev':
            topk_idx = topk_ids_b
        elif self.m == 'fpv':
            topk_idx = topk_ids_f
        else:
            topk_idx = torch.tensor(list(set(topk_ids_b.tolist() + topk_ids_f.tolist())))# 将bev和fpv的车道线合并

        topk_l_b = f_lane_b[i][topk_idx] if f_lane_b is not None else None
        topk_l_f = f_lane_f[i][topk_idx] if f_lane_f is not None else None
        return topk_l_b, topk_l_f

    def get_2D_hidden(self, i, goals_2D, inputs, hidden_state, inputs_lengths, topk_lanes, tag):
        '''
            这个函数是用来计算2D目标的特征的
            i: 第i个样本
            goals_2D: 2D目标的坐标
            inputs: 输入
            hidden_state: 隐藏状态
            inputs_lengths: 输入的长度
            topk_lanes: 车道线
            tag: 标签
        '''
        if self.cfg.MODEL.share_weight:# 如果共享权重
            cross_attn = self.goals_2D_cross_attention 
            sub_graph = self.goals_2D_point_sub_graph
            goals_decoder = self.stage_one_goals_2D_decoder
        else:
            if tag == 'bev':
                cross_attn = self.goals_2D_cross_attention_b
                sub_graph = self.goals_2D_point_sub_graph_b
                goals_decoder = self.stage_one_goals_2D_decoder_b
            else:
                cross_attn = self.goals_2D_cross_attention_f
                sub_graph = self.goals_2D_point_sub_graph_f
                goals_decoder = self.stage_one_goals_2D_decoder_f

        goals_2D_tensor = torch.tensor(goals_2D, dtype=torch.float, device=self.device)# [n_goal,2]，初始化目标的2D坐标
        goals_2D_hidden = sub_graph(goals_2D_tensor.unsqueeze(0), hidden_state[0:1, :]).squeeze(0)  # [n_goal,2] -> [n_goal,128]
        goals_2D_hidden_attention = cross_attn(
            goals_2D_hidden.unsqueeze(0), inputs[i][:inputs_lengths[i]].unsqueeze(0)
        ).squeeze(0)  # [n_goal,128], [n_lane+n_pen,128] -> [n_goal,128] -> [n_goal,128]，线性编码

        stage_one_goals_2D_hidden_attention = cross_attn(
            goals_2D_hidden.unsqueeze(0), topk_lanes.unsqueeze(0)).squeeze(0)# [n_goal,128], [n_lane,128] -> [n_goal,128] -> [n_goal,128]，线性编码
        li = [hidden_state[0, :].unsqueeze(0).expand(goals_2D_hidden.shape),
              goals_2D_hidden, goals_2D_hidden_attention, stage_one_goals_2D_hidden_attention]# li是一个列表，包含了4个张量，分别是隐藏状态，2D目标的特征，2D目标的注意力，2D目标和车道线的注意力

        feature_2D = goals_decoder(torch.cat(li, dim=-1))# [n_goal,128]，解码获取特征

        # scores = scores.squeeze(-1)
        # scores.masked_fill(goals_mask, -np.inf)
        # scores = F.log_softmax(scores, dim=-1)

        return feature_2D

    def gate_score(self, feature, tag):
        '''
            这个函数是用来计算门控得分的，门控得分是用来计算2D目标的得分的
            feature: 特征
            tag: 标签
        '''
        # feature=[n_goals,128]
        if tag == 'bev':
            score_mlp = self.score_mlp_b
        else:
            score_mlp = self.score_mlp_f

        res = score_mlp(feature)  # [n_goals,2]
        '''
            res[:, 0]是得分，res[:, 1]是门控得分，norm_act是sigmoid
        '''
        return res[:, 0], torch.exp(self.norm_act(res[:, 1]))

    def get_3D_scores(self, goals_3D, infer, inputs_b, inputs_f, h_state_b, h_state_f, lens_b, lens_f,
                      i, mapping_b, mapping_f, topk_l_b, topk_l_f):
        '''
            这个函数是用来计算3D目标的得分的
            goals_3D: 3D目标的坐标
            infer: 推断
            inputs_b: bev的输入
            inputs_f: fpv的输入
            h_state_b: bev的隐藏状态
            h_state_f: fpv的隐藏状态
            lens_b: bev的长度
            lens_f: fpv的长度
            i: 第i个样本
            mapping_b: bev的映射
            mapping_f: fpv的映射
            topk_l_b: bev的车道线
            topk_l_f: fpv的车道线
        '''
        # goals_3D=np[n_goals,3]
        goals_2D_b, masks_b = trans_3D_2D(self.calib, i, mapping_b, goals_3D, 'bev')  # [n_goals,2], [n_goals], calib是相机的标定参数
        # goals_2D_f, masks_f = trans_3D_2D(self.calib, i, mapping_b, goals_3D, 'fpv')  # [n_goals,2], [n_goals] 原代码貌似是写错了，应该是mapping_f
        goals_2D_f, masks_f = trans_3D_2D(self.calib, i, mapping_f, goals_3D, 'fpv')

        if (self.m == 'both') or (self.m == 'bev'):
            feature_2D_b = self.get_2D_hidden(i, goals_2D_b, inputs_b, h_state_b, lens_b, topk_l_b, 'bev')  # [n_goals,128]
        else:
            feature_2D_b = None
        if (self.m == 'both') or (self.m == 'fpv'):
            feature_2D_f = self.get_2D_hidden(i, goals_2D_f, inputs_f, h_state_f, lens_f, topk_l_f, 'fpv')  # [n_goals,128]
        else:
            feature_2D_f = None

        if not infer:  # random mask mask=true是不要的, 随机遮挡，用于训练
            rand_b = np.random.rand(masks_b.shape[0])  # 均匀分布
            masks_b = masks_b | (rand_b < self.cfg.MODEL.random_mask)  # 有一个true，就为true
            rand_f = np.random.rand(masks_f.shape[0])
            masks_f = masks_f | (rand_f < self.cfg.MODEL.random_mask)
        masks_tensor_b = torch.tensor(masks_b, dtype=torch.bool, device=self.device)# 创建一个和masks_b相同形状的空张量，用于遮挡
        masks_tensor_f = torch.tensor(masks_f, dtype=torch.bool, device=self.device)

        # cross
        if self.cfg.MODEL.cross_type == 0:  # not cross，是指bev和fpv的特征不交叉
            feature_2D = feature_2D_b if feature_2D_b is not None else feature_2D_f # 如果bev的特征不为空，就用bev的特征，否则用fpv的特征
            scores_2D = self.score_mlp(feature_2D).squeeze(-1)  # [n_goals] # [n_goals,1] -> [n_goals]
        elif self.cfg.MODEL.cross_type == 1:
            # 第一种方案，直接mask相加
            # unsqueeze(1)将masks_tensor_b的维度从[n_goals]变成[n_goals,1]，repeat(1, self.hidden_size)将masks_tensor_b的维度从[n_goals]变成[n_goals,128]
            # masked_fill将feature_2D_b中的masks_tensor_b为true的位置设置为0，mask用于dropout
            feature_2D_b = torch.masked_fill(feature_2D_b, masks_tensor_b.unsqueeze(1).repeat(1, self.hidden_size), 0)
            feature_2D_f = torch.masked_fill(feature_2D_f, masks_tensor_f.unsqueeze(1).repeat(1, self.hidden_size), 0)

            feature_2D = feature_2D_b + feature_2D_f
            scores_2D = self.score_mlp(feature_2D).squeeze(-1)  # [n_goals]
        elif self.cfg.MODEL.cross_type == 2:
            # 第二种方案，mask concate
            feature_2D_b = torch.masked_fill(feature_2D_b, masks_tensor_b.unsqueeze(1).repeat(1, self.hidden_size), 0)
            feature_2D_f = torch.masked_fill(feature_2D_f, masks_tensor_f.unsqueeze(1).repeat(1, self.hidden_size), 0)

            feature_2D = torch.cat([feature_2D_b, feature_2D_f], dim=-1)
            scores_2D = self.score_mlp(feature_2D).squeeze(-1)  # [n_goals]
        elif self.cfg.MODEL.cross_type == 3:
            # gate fusion
            feature_2D_b = torch.masked_fill(feature_2D_b, masks_tensor_b.unsqueeze(1).repeat(1, self.hidden_size), 0)
            feature_2D_f = torch.masked_fill(feature_2D_f, masks_tensor_f.unsqueeze(1).repeat(1, self.hidden_size), 0)

            s_bev, g_bev = self.gate_score(feature_2D_b, 'bev') # [n_goals], [n_goals] s_bev是bev的得分，g_bev是bev的门控得分
            s_fpv, g_fpv = self.gate_score(feature_2D_f, 'fpv')
            scores_2D = g_bev / (g_bev + g_fpv) * s_bev + g_fpv / (g_bev + g_fpv) * s_fpv  # [n_goals]

        else:
            scores_2D = torch.ones(goals_3D.shape[0])# [n_goals]，全1，不遮挡，用于推理

        scores_2D = F.log_softmax(scores_2D, dim=-1)  # [n_goals]

        return scores_2D  # [n_goals]

    def get_dense_goals(self, i, goals_3D, scores, mapping_b, mapping_f):
        '''
            这个函数是用来稀疏点云数据稠密化的
            i: 第i个样本
            goals_3D: 3D目标的坐标
            scores: 得分
            mapping_b: bev的映射
            mapping_f: fpv的映射
            投影与反向投影：使用摄像头内外参将3D点投影到FPV图像平面上，再利用反向投影来还原3D空间中的位置信息。这种方式可以在图像上标注目标，但受视角限制，无法完全覆盖所有目标。
            多视角融合：结合BEV和FPV视角的信息，利用BEV处理全局目标分配，FPV则用于细粒度的目标分析，特别是在车辆前方的重点区域。
            深度估计：通过FPV图像的深度信息估算3D坐标，结合3D点云数据处理，但需要额外的计算量。
        '''
        k = 150 # k是最大的目标数
        _, topk_ids = torch.topk(scores, k=min(k, len(scores)))# 获取得分最高的k个目标
        topk_ids = topk_ids.detach().cpu().numpy()

        goals_3D_dense_ = get_neighbour_3Dpoints(goals_3D[topk_ids], topk_ids=topk_ids)# 稀疏点云数据稠密化，topk_ids是最高得分的目标，但是在函数里没有用到

        goals_3D_dense = np.concatenate([goals_3D_dense_, goals_3D], axis=0)  # np[d_goals,3]，将稠密化的目标和原始的目标拼接，一种残差的思想

        '''
            序列中目标的最后一步的索引，通常代表未来轨迹中目标将要到达的最终位置。在预测任务中，模型可能需要在稠密化的候选点集合中找到一个距离真实终点位置最近的候选目标点，用于进一步校准模型预测的输出。
            get_dis 函数用于计算稠密目标集合 goals_3D_dense 中各点与 label[final_idx]（真实终点坐标）的距离。
            np.argmin 返回距离最小的点的索引，即在稠密集合中找到最接近真实目标位置的候选点。
        '''
        label = np.array(mapping_b[i]['labels']).reshape([self.future_frame_num, 2])# [future_frame_num,2]，获取真实的目标坐标
        final_idx = mapping_b[i].get('final_idx', -1)# 获取最终的索引
        mapping_b[i]['goals_2D_labels'] = np.argmin(get_dis(goals_3D_dense, label[final_idx])) # 计算稠密化的目标和真实目标的距离，取最小的距离

        return goals_3D_dense  # np[d_goals,3]

    def hidden_reinforce(self, i, h_state_b, h_state_f, topk_l_b, topk_l_f):
        '''
            这个函数是用来增强隐藏状态的
            i: 第i个样本
            h_state_b: bev的隐藏状态
            h_state_f: fpv的隐藏状态
            topk_l_b: bev的车道线
            topk_l_f: fpv的车道线
            h_state_b[0:1, :]可以代表初始状态或全局背景信息
        '''
        if self.cfg.MODEL.self_rein:
            if (self.m == 'both') or (self.m == 'bev'):
                self_hidden_b = self.reinforce_attention_b(
                    h_state_b[0:1, :].unsqueeze(0), topk_l_b.unsqueeze(0)
                ).squeeze(0)# [1,128]，[n_lane,128] -> [1,128] -> [128], h_state_b[0:1, :]是取第一个时刻的隐藏状态,作为Q，topk_l_b是车道线的特征，作为K和V，# 我感觉这里也写反了

                # self_hidden_b = self.reinforce_attention_b(
                #     topk_l_b.unsqueeze(0), h_state_b[0:1, :].unsqueeze(0)
                # ).squeeze(0)
            else:
                self_hidden_b = None

            if (self.m == 'both') or(self.m == 'fpv'):
                self_hidden_f = self.reinforce_attention_f(
                    h_state_f[0:1, :].unsqueeze(0), topk_l_f.unsqueeze(0)
                ).squeeze(0)
                # self_hidden_f = self.reinforce_attention_f(
                #     topk_l_f.unsqueeze(0), h_state_f[0:1, :].unsqueeze(0)
                # ).squeeze(0)
            else:
                self_hidden_f = None
            return self_hidden_b, self_hidden_f
        else:
            self_hidden_b = h_state_b[0:1, :] if h_state_b is not None else None
            self_hidden_f = h_state_f[0:1, :] if h_state_f is not None else None
            return self_hidden_b, self_hidden_f

    def goals_3D_example(self, i, goals_3D, mapping_b, mapping_f, f_lane_b, f_lane_f, l_masks_b, l_masks_f,
                         inputs_b, inputs_f, lens_b, lens_f, h_states_b, h_states_f, loss_b, loss_f, infer):
        '''
            这个函数是用来计算3D目标的得分的
            i: 第i个样本
            goals_3D: 3D目标的坐标
            mapping_b: bev的映射
            mapping_f: fpv的映射
            f_lane_b: bev的车道线特征
            f_lane_f: fpv的车道线特征
            l_masks_b: bev的mask
            l_masks_f: fpv的mask
            inputs_b: bev的输入
            inputs_f: fpv的输入
            lens_b: bev的长度
            lens_f: fpv的长度
            h_states_b: bev的隐藏状态, [n_ped,t,128], n_ped是需要预测的目标数
            h_states_f: fpv的隐藏状态, [n_ped,t,128]
            loss_b: bev的loss
            loss_f: fpv的loss
            infer: 推断
        '''
        h_state_b = h_states_b[i] if h_states_b is not None else None
        h_state_f = h_states_f[i] if h_states_f is not None else None
        # 通过隐藏状态和车道线的特征，计算车道线的得分
        topk_l_b, topk_l_f = self.lane_scoring(
            i, mapping_b, mapping_f, loss_b, loss_f, f_lane_b, f_lane_f, l_masks_b, l_masks_f,
            inputs_b, inputs_f, lens_b, lens_f, h_state_b, h_state_f
        )  # [?,128]
        # 通过隐藏状态和车道线得分，增强隐藏状态，但是如果是将车道线得分作为K和V，那么Q就是隐藏状态，这样返回的是注意后的车道线得分
        h_state_b, h_state_f = self.hidden_reinforce(i, h_state_b, h_state_f, topk_l_b, topk_l_f)
        
        get_scores_inputs = (inputs_b, inputs_f, h_state_b, h_state_f, lens_b, lens_f, i,
                             mapping_b, mapping_f, topk_l_b, topk_l_f)
        '''
            先预计算3D目标的得分，然后通过得分最高的目标，计算稠密化的3D目标，再通过稠密化的3D目标，计算得分最高的目标
        '''
        # 通过3D目标的坐标，计算3D目标的得分
        scores = self.get_3D_scores(goals_3D, infer, *get_scores_inputs)  # [n_goals]
        # 通过3D目标的坐标，计算稠密化的3D目标
        goals_3D_dense = self.get_dense_goals(i, goals_3D, scores, mapping_b, mapping_f)  # np[d_goals,3]
        # 通过稠密化的3D目标的坐标，计算得分最高的目标得分
        scores = self.get_3D_scores(goals_3D_dense, infer, *get_scores_inputs)  # [d_goals]

        gt_points = mapping_b[i]['labels'].reshape([self.future_frame_num, 2])  # [30,2]
        labels_is_valid = mapping_b[i]['labels_is_valid']
        highest_index = torch.argmax(scores).item()# 获取得分最高的目标的索引

        if infer:
            mapping_b[i]['vis.goals_3D'] = goals_3D_dense
            mapping_b[i]['vis.scores'] = np.array(scores.tolist())
            # mapping_b[i]['vis.labels'] = gt_points
            # mapping_b[i]['vis.labels_is_valid'] = labels_is_valid

        return scores, goals_3D_dense, highest_index

    def forward(self, batch_size, mapping_b, mapping_f, f_lane_b, f_lane_f, l_masks_b, l_masks_f,
                inputs_b, inputs_f, lens_b, lens_f, h_states_b, h_states_f, infer):
        loss_b = torch.zeros(batch_size, device=self.device)
        loss_f = torch.zeros(batch_size, device=self.device)

        scores_list = []
        goals_3D_list = []
        highest_list = []

        # 编码阶段的cross attention
        if self.cfg.MODEL.cross_enc:
            res_b = h_states_b
            res_f = h_states_f
            h_states_b = self.b2f_cross_attn(res_b, res_f) + res_b # 我感觉这里写反了，应该是拿res_b作为K和V，res_f作为Q
            h_states_f = self.f2b_cross_attn(res_f, res_b) + res_f
            # h_states_b = self.b2f_cross_attn(res_f, res_b) + res_b # 这样写才对
            # h_states_f = self.f2b_cross_attn(res_b, res_f) + res_f

        for i in range(batch_size):
            goals_2D = mapping_b[i]['goals_2D']  # [s_goals,2]
            goals_3D = np.concatenate([goals_2D, np.zeros([goals_2D.shape[0], 1])], axis=-1)  # [s_goals,3]

            scores, goals_3D_dense, highest_index = self.goals_3D_example(
                i, goals_3D, mapping_b, mapping_f, f_lane_b, f_lane_f, l_masks_b, l_masks_f,
                inputs_b, inputs_f, lens_b, lens_f, h_states_b, h_states_f, loss_b, loss_f, infer
            )  # [d_goals], np[d_goals,3], np[3]

            scores_list.append(scores)  # [d_goals]
            # goals_dense_b, _ = trans_3D_2D(self.calib, i, mapping_b, goals_3D_dense, 'bev')
            # goals_dense_f, _ = trans_3D_2D(self.calib, i, mapping_f, goals_3D_dense, 'fpv')
            # goals_l_b.append(goals_dense_b)
            # goals_l_f.append(goals_dense_f)
            goals_3D_list.append(goals_3D_dense)  # np[d_goals,2]
            highest_list.append(highest_index)  # int

        return loss_b, loss_f, scores_list, goals_3D_list, highest_list





