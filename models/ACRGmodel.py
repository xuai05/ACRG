from __future__ import division

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_util import norm_col_init, weights_init

from utils.data_utils import sarpn_depth_h5

from .model_io import ModelOutput
from .transformer import Transformer, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, \
    TransformerDecoder


import matplotlib.pyplot as plt
import numpy as np

class VisualTransformer(Transformer):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=256, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super(VisualTransformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, query_embed):
        bs, n, c = src.shape
        # print(src.size())
        src = src.permute(1, 0, 2)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src)
        hs = self.decoder(tgt, memory, query_pos=query_embed)
        return hs.transpose(0, 1), memory.permute(1, 2, 0).view(bs, c, n)

class ACRGModel(nn.Module):
    def __init__(self, args):
        super(ACRGModel, self).__init__()
        # visual representation part
        # the networks used to process local visual representation should be replaced with linear transformer
        self.num_cate = args.num_category
        self.image_size = 300

        # global visual representation learning networks
        resnet_embedding_sz = 512
        hidden_state_sz = args.hidden_state_sz
        self.global_conv = nn.Conv2d(resnet_embedding_sz, 256, 1)
        self.global_pos_embedding = get_gloabal_pos_embedding(7, 128)
        
        

        # previous action embedding networks
        action_space = args.action_space
        self.embed_action = nn.Linear(action_space, 64)

        self.visual_transformer = VisualTransformer(
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
        )

        self.visual_rep_embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=args.dropout_rate)
        )

        # ==================================================
        # navigation policy part
        self.lstm_input_sz = 3200
        self.hidden_state_sz = hidden_state_sz

        self.lstm = nn.LSTM(self.lstm_input_sz, hidden_state_sz, 2)

        self.critic_linear_1 = nn.Linear(hidden_state_sz, 64)
        self.critic_linear_2 = nn.Linear(64, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, action_space)

        # ==================================================
        # weights initialization
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.global_conv.weight.data.mul_(relu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear_1.weight.data = norm_col_init(
            self.critic_linear_1.weight.data, 1.0
        )
        self.critic_linear_1.bias.data.fill_(0)
        self.critic_linear_2.weight.data = norm_col_init(
            self.critic_linear_2.weight.data, 1.0
        )
        self.critic_linear_2.bias.data.fill_(0)

        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_ih_l1.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)
        self.lstm.bias_hh_l1.data.fill_(0)


        self.graph_detection_other_info_linear_1 = nn.Linear(7, 100)
        self.graph_detection_other_info_linear_2 = nn.Linear(100, 100)
        self.graph_detection_other_info_linear_3 = nn.Linear(100, 100)
        self.graph_detection_other_info_linear_4 = nn.Linear(100, 100)
        self.graph_detection_other_info_linear_5 = nn.Linear(100, 100)
        self.graph_detection = nn.Linear(262, 256)


        self.graph_relation_x = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        

        self.graph_relation_y = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.graph_relation_d = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.graph_relation_x_1 = nn.Linear(6, 100)
        self.graph_relation_y_1 = nn.Linear(4, 100)
        self.graph_relation_depth_1 = nn.Linear(3, 100)

        self.graph_relation_x_group = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )
        self.graph_relation_y_group = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )
        self.graph_relation_depth_group = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )


        self.depth_linear = nn.Sequential(
            nn.Linear(152, 114),
            nn.ReLU(),
            )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, 128, 5),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=2),
            nn.Conv2d(128, 256, 5),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7,7))
            )
        self.TDorg_graph_detection = nn.Linear(2, 1)
        
        
        # 可删除
        self.local_embedding = nn.Sequential(
                nn.Linear(256, 249),
                nn.ReLU(),
            )


    def embedding(self, state, detection_inputs, action_embedding_input,state_name,scene_name):
        # detection_inputs contains the features embedding [100, 256], the scores [100], the labels [100], teh bboxes [100,4] is the Center coordinates and height width,the indicator [100,1]
        # what is the target [9],tensor([-2.4938e+35,  4.5818e-41, -1.6821e-19,  3.0966e-41,  1.4013e-45, 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00], device='cuda:0')????

        depth = sarpn_depth_h5(str(state_name),str(scene_name)).squeeze(dim=0).squeeze(dim=0)
        for i in range(len(detection_inputs['indicator'])):
            if detection_inputs['indicator'][i] == 1:
                target_bbox = detection_inputs['bboxes'][i]    
                detection_inputs["x_dim"][i] = torch.tensor([target_bbox[1],target_bbox[3]])
                detection_inputs["y_dim"][i] = torch.tensor([target_bbox[0],target_bbox[2]]) 
                int_target0, int_target1, int_target2, int_target3  = int((target_bbox[0]*152)/300),int((target_bbox[1]*114)/300),int((target_bbox[2]*152)/300),int((target_bbox[3]*114)/300) 
                if int_target0<=0:int_target0=0
                if int_target1<=0:int_target1=0
                if int_target2<=0:int_target2=0
                if int_target3<=0:int_target3=0
                if int_target0 == int_target2:      #Prevent nan
                    if int_target0 == 0:
                        int_target2 = int_target2 + 1
                    else:
                        int_target0 = int_target0-1
                if int_target1 == int_target3: 
                    if int_target1 == 0:
                        int_target3 = int_target3 + 1
                    else:
                        int_target1 = int_target1-1
                
                detection_inputs["depth"][i] = depth[int_target1:int_target3,int_target0:int_target2].mean()

                if detection_inputs["depth"][i] != detection_inputs["depth"][i]:
                    print("int_target0, int_target1, int_target2, int_target3 is {},{},{},{}".format(int_target0, int_target1, int_target2, int_target3))
                    print("detection_inputs['depth'][i] is {}".format(detection_inputs["depth"][i]))
                break
        detection_inputs['labels'] = detection_inputs['labels'].unsqueeze(dim=1)
        target_x_info = torch.cat((
            detection_inputs['labels'], 
            detection_inputs["x_dim"], 
            detection_inputs["y_dim"], 
            detection_inputs['indicator']), dim=1) 
        # target_y_info = torch.cat((
        #     detection_inputs['labels'], 
        #     detection_inputs["y_dim"], 
        #     detection_inputs['indicator']), dim=1) 
        target_depth_info = torch.cat((
            detection_inputs['labels'], 
            detection_inputs["depth"], 
            detection_inputs['indicator']), dim=1)

        target_x_info = F.relu(self.graph_relation_x_1(target_x_info))
        target_x_info = target_x_info.t()               # .t() is transpose()
        target_x_info = F.relu(self.graph_relation_x_group(target_x_info))

        # target_y_info = F.relu(self.graph_relation_y_1(target_y_info))
        # target_y_info = target_y_info.t()               # .t() is transpose()
        # target_y_info = F.relu(self.graph_relation_y_group(target_y_info))

        target_depth_info = F.relu(self.graph_relation_depth_1(target_depth_info))
        target_depth_info = target_depth_info.t()               # .t() is transpose()
        target_depth_info = F.relu(self.graph_relation_depth_group(target_depth_info))
        
        Tdorg_feature = torch.cat((target_x_info.unsqueeze(dim=2),target_depth_info.unsqueeze(dim=2)),dim=2)
        ######

        
        Tdorg_feature = F.relu(self.TDorg_graph_detection(Tdorg_feature)).squeeze(dim=2)

        # 100 256 特征 * 100，100
        detection_input = torch.mm(detection_inputs['features'].t(), Tdorg_feature).t()      # employ ORG as an attention map to encode the local appearance features. [100, 256]
        detection_input = torch.cat((
            detection_input,
            detection_inputs['bboxes'], 
            detection_inputs['labels'], 
            detection_inputs['indicator']), dim=1).unsqueeze(dim=0)
        detection_input = F.relu(self.graph_detection(detection_input))              #Completely follow ORG code logic   detection_input shape is torch.Size([1, 100, 256])


        # state shape [1,512,7,7]
        image_embedding = F.relu(self.global_conv(state))       #get the gobal_feature
        gpu_id = image_embedding.get_device()

        image_embedding = image_embedding + self.global_pos_embedding.cuda(gpu_id)
    
        
        image_embedding = image_embedding.reshape(1, -1, 49)        #reshape to hw*d = 49*256

        visual_queries = image_embedding
        visual_representation, encoded_rep = self.visual_transformer(src=detection_input,
                                                                        query_embed=visual_queries)
        out = self.visual_rep_embedding(visual_representation)          #[1, 49, 64]

        action_embedding = F.relu(self.embed_action(action_embedding_input)).unsqueeze(dim=1)
        out = torch.cat((out, action_embedding), dim=1)         # #[1, 50, 64]
       
        out = out.reshape(1, -1)

        return out, image_embedding

    def a3clstm(self, embedding, prev_hidden_h, prev_hidden_c):
        embedding = embedding.reshape([1, 1, self.lstm_input_sz])
        output, (hx, cx) = self.lstm(embedding, (prev_hidden_h, prev_hidden_c))
        x = output.reshape([1, self.hidden_state_sz])

        actor_out = self.actor_linear(x)
        critic_out = self.critic_linear_1(x)
        critic_out = self.critic_linear_2(critic_out)

        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):
        state = model_input.state
        # state is torch.Size([1, 512, 7, 7])
        (hx, cx) = model_input.hidden

        detection_inputs = model_input.detection_inputs
        action_probs = model_input.action_probs
        # modify add
        state_name = model_input.state_name
        scene_name = model_input.scene_name

        x, image_embedding = self.embedding(state, detection_inputs, action_probs,state_name,scene_name)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, hx, cx)

        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
        )

#position embedding
def get_gloabal_pos_embedding(size_feature_map, c_pos_embedding):
    mask = torch.ones(1, size_feature_map, size_feature_map)

    y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(2, dtype=torch.float32)

    dim_t = torch.arange(c_pos_embedding, dtype=torch.float32)

    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / c_pos_embedding)           #torch==1.9
    # dim_t = 10000 ** (2 * (dim_t // 2) / c_pos_embedding)           # torch == 1.4

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos


