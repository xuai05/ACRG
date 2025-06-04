""" Contains the Episodes for Navigation. """
import random
import sys
from time import time
import torch
import numpy as np
import pandas as pd

from datasets.constants import GOAL_SUCCESS_REWARD, STEP_PENALTY, DUPLICATE_STATE, UNSEEN_STATE
from datasets.constants import DONE
from datasets.environment import Environment

from utils.model_util import gpuify, toFloatTensor
from utils.action_util import get_actions
from utils.model_util import gpuify
from .episode import Episode

from utils.data_utils import sarpn_depth_h5


class BasicEpisode(Episode):
    """ Episode for Navigation. """

    def __init__(self, args, gpu_id, strict_done=False):
        super(BasicEpisode, self).__init__()

        self._env = None

        self.gpu_id = gpu_id
        self.strict_done = strict_done
        self.task_data = None
        self.glove_embedding = None
        self.actions = get_actions(args)
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self._last_action_embedding_idx = 0
        self.target_object = None
        self.prev_frame = None
        self.current_frame = None
        self.scene = None

        self.scene_states = []
        if args.eval:
            random.seed(args.seed)

        self._episode_times = 0
        self.seen_percentage = 0

        self.state_reps = []
        self.state_memory = []
        self.action_memory = []
        self.obs_reps = []

        self.episode_length = 0
        self.target_object_detected = False

        # tools
        self.states = []
        self.actions_record = []
        self.action_outputs = []
        self.detection_results = []

        # imitation learning
        # self.imitation_learning = args.imitation_learning
        self.action_failed_il = False

        self.action_probs = []

        # self.meta_learning = args.update_meta_network
        self.meta_predictions = []

        self.warm_up = args.warm_up
        self.num_workers = args.num_workers
        self.episode_num = 0


        

    @property
    def environment(self):
        return self._env

    @property
    def actions_list(self):
        return [{"action": a} for a in self.actions]

    @property
    def episode_times(self):
        return self._episode_times

    @episode_times.setter
    def episode_times(self, times):
        self._episode_times = times

    def reset(self):
        self.done_count = 0
        self.duplicate_count = 0
        self._env.back_to_start()

    def state_for_agent(self):
        return self.environment.current_frame

    def current_detection_feature(self):
        return self.environment.current_detection_feature

    def current_agent_position(self):
        """ Get the current position of the agent in the scene. """
        return self.environment.current_agent_position

    def step(self, action_as_int, model_input, memory):
        action = self.actions_list[action_as_int]
        if action["action"] != DONE:
            self.environment.step(action)
        else:
            self.done_count += 1

        reward, terminal, action_was_successful = self.judge(action,model_input,memory)    
        return reward, terminal, action_was_successful

    def judge(self, action, model_input, memory):
        """ Judge the last event. """
        reward = STEP_PENALTY
        # Thresholding replaced with simple look up for efficiency.
        if self.environment.controller.state in self.scene_states:
            if action["action"] != DONE:
                if self.environment.last_action_success:
                    self.duplicate_count += 1
                else:
                    self.failed_action_count += 1
        else:
            self.scene_states.append(self.environment.controller.state)
        done = False

        #对于RL层面根据深度信息异常进行避障和前进奖励机制的设定
        RLavoid = False
        if model_input.trainflag and RLavoid:
            print("RLavoid")
                #  modify
            #  doctor wang
            #  add model_input
            target = None
            for i in range(len(model_input.detection_inputs['indicator'])):
                if model_input.detection_inputs['indicator'][i] == 1:
                    target = model_input.detection_inputs['bboxes'][i]                  # The detection frame position where the target appears target is tensor([ 57.8484, 194.4995, 111.0906, 288.8900], device='cuda:5')
                    break
            
            if target is not None:          #save current frame
                # aera = (target[2]-target[0])*(target[3]-target[1])                  # Target area size
                # According to the position of the target detection frame in the corresponding depth map, the average depth of the target position is given
                depth = sarpn_depth_h5(str(model_input.state_name),str(model_input.scene_name)).squeeze(dim=0).squeeze(dim=0)
                int_target0, int_target1, int_target2, int_target3  = int((target[0]*114)/300),int((target[1]*152)/300),int((target[2]*114)/300),int((target[3]*152)/300)            # 0,2 is row index 1,3 is column index

                # Judging target depth
                # paln 1:box mean
                target_depth = depth[int_target0:int_target2,int_target1:int_target3]
                target_mean_depth = target_depth.mean()         # average depth of the target position
                # print("target_mean_depth is {} ".format(target_mean_depth))

                # Judgment area selection .The range for calculating whether there is a sudden change still needs to be set specifically???
                # plan 1: Three divisions
                if int_target3 < 76:
                    mutat_deter_range = depth[int_target0:114,0:76]
                elif int_target1 > 76:
                    mutat_deter_range = depth[int_target0:114,76:152]
                else:
                    mutat_deter_range = depth[int_target0:300,38:114]
                mutat_deter_list = torch.mean(mutat_deter_range, dim=1)     #Average by row shape

                # Judging the mutation 
                mutat_flag = 0
                # For each plan, can view the saved pictures and then make a manual judgment on the depth map, and observe which plan has better judgment obstacles
                # plan 1 : Simple threshold
                    # threshold = 0.25
                    # j = mutat_deter_list[0]
                    # for i in mutat_deter_list[1:]:
                    #     if i - j > threshold:
                    #         mutat_flag = 1
                    #         break
                    # print("mutat_flag is {}".format(mutat_flag))

                # Assuming that the trend of change is uniform,
                # extract step_list and step_mean
                step_list = mutat_deter_list[:-1] - mutat_deter_list[1:]
                step_mean = torch.mean(step_list)
                # plan 2: 3 sigmoid theory:
                step_std = torch.std(step_list)
                for i in step_list:
                    if i > step_mean + 3 * step_std or i < step_mean - 3 * step_std:
                        mutat_flag = 1
                        break
                aviod_r, move_r = 0,0
                if not len(memory) < 1:
                    if memory[-1]["mutat_flag"] == 1 and mutat_flag == 0: aviod_r = 0.2
                    if memory[-1]["mutat_flag"] == 0 and mutat_flag == 0: 
                        if memory[-1]["target_mean_depth"] > target_mean_depth: move_r = 0.1

                Inter_memory = {"mutat_flag":mutat_flag,"target_mean_depth":target_mean_depth,}
                memory.append(Inter_memory)
                reward = aviod_r + move_r

        if action["action"] == DONE:
            action_was_successful = False
            for id_ in self.task_data:
                if self.environment.object_is_visible(id_):
                    reward = GOAL_SUCCESS_REWARD
                    done = True
                    action_was_successful = True
                    break
        else:
            action_was_successful = self.environment.last_action_success

        return reward, done, action_was_successful

    # Set the target index.
    @property
    def target_object_index(self):
        """ Return the index which corresponds to the target object. """
        return self._target_object_index

    @target_object_index.setter
    def target_object_index(self, target_object_index):
        """ Set the target object by specifying the index. """
        self._target_object_index = gpuify(
            torch.LongTensor([target_object_index]), self.gpu_id
        )

    def _new_episode(self, args, scenes, targets):
        """ New navigation episode. """
        scene = random.choice(scenes)               #scene is FloorPlanX
        self.scene = scene

        if self._env is None:
            # load scene data
            # modify
            self._env = Environment(
                offline_data_dir=args.data_dir,
                use_offline_controller=True,
                grid_size=0.25,
                detection_feature_file_name=args.detection_feature_file_name,
                images_file_name=args.images_file_name,
                visible_object_map_file_name=args.visible_map_file_name,
                optimal_action_file_name=args.optimal_action_file_name,
            )
            self._env.start(scene)
        else:
            self._env.reset(scene)

        self.task_data = []
        objects = self._env.all_objects()       # list of object obtain: Towel|-01.62|+01.40|+01.80
        visible_objects = [obj.split("|")[0] for obj in objects]
        intersection = [obj for obj in visible_objects if obj in targets]  
        # Randomly select a target from the objects visible in the current scene
        idx = random.randint(0, len(intersection) - 1)
        goal_object_type = intersection[idx]
        self.target_object = goal_object_type    

        for id_ in objects:
            type_ = id_.split("|")[0]
            if goal_object_type == type_:
                self.task_data.append(id_)


        warm_up_path_len = 200
        if (self.episode_num * self.num_workers) < 500000:
            warm_up_path_len = 5 * (int((self.episode_num * self.num_workers) / 50000) + 1)
        else:
            self.warm_up = False

        if self.warm_up:
            for _ in range(10):
                self._env.randomize_agent_location()
                shortest_path_len = 1000
                for _id in self.task_data:
                    path_len = self._env.controller.shortest_path_to_target(self._env.start_state, _id)[1]
                    if path_len < shortest_path_len:
                        shortest_path_len = path_len
                if shortest_path_len <= warm_up_path_len:
                    break
        else:
            self._env.randomize_agent_location()


    def new_episode(self, args, scenes, targets):
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self.episode_length = 0
        self.prev_frame = None
        self.current_frame = None
        self.scene_states = []

        self.state_reps = []
        self.state_memory = []
        self.action_memory = []

        self.target_object_detected = False

        self.episode_times += 1
        self.episode_num += 1

        self.states = []
        self.actions_record = []
        self.action_outputs = []
        self.detection_results = []
        self.obs_reps = []

        self.action_failed_il = False

        self.action_probs = []
        self.meta_predictions = []

        self._new_episode(args, scenes, targets)



