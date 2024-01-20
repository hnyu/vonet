# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# requires tensorflow 2.4.1 and tensorflow_datasets 4.7
import gym
from gym import spaces

import os
import matplotlib.cm
import cv2
from typing import Union, List
import glob

import numpy as np
import pickle
from skimage import color


class MOVi(gym.Env):
    """An environment for loading the MOVi datasets
    `<https://github.com/google-research/kubric/tree/main/challenges/movi>`_.

    The datasets are expected to be preprocessed first as local pickle files. See
    `scripts/download_movi_data.py` for details.
    """
    SIZE = 128

    def __init__(self,
                 dataset_root: Union[str, List[str]],
                 n_videos: int = None,
                 test: bool = False,
                 test_video_path: str = None):
        """
        Args:
            dataset: the name of the dataset. Should be one of [
                "movi_a", "movi_b", "movi_c", "movi_d", "movi_e"].
        """
        obs_space = {
            "rgb":
                spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.SIZE, self.SIZE, 3),
                    dtype=np.uint8),
            "segmentation":
                spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.SIZE, self.SIZE, 1),
                    dtype=np.uint8)
        }
        self.observation_space = spaces.Dict(obs_space)
        self.action_space = spaces.Discrete(2)

        self._training_videos = []
        self._val_videos = []
        if not isinstance(dataset_root, list):
            dataset_root = [dataset_root]
        for dr in dataset_root:
            tr = glob.glob(os.path.join(dr, "train/*.pkl"))
            val = glob.glob(os.path.join(dr, "validation/*pkl"))
            self._training_videos.extend(tr)
            self._val_videos.extend(val)

        if n_videos is not None:
            n_val_videos = int(n_videos * 0.025)
            n_train_videos = n_videos - n_val_videos
            self._training_videos = self._training_videos[:n_train_videos]
            self._val_videos = self._val_videos[:n_val_videos]

        print("Total training videos: ", len(self._training_videos))
        print("Total validation videos: ", len(self._val_videos))

        self._test = test
        if test:
            print("!!!!!Test mode!!!!!")
        self._test_n = 0
        self._step = 0
        self._video = None
        self._video_id = None
        self._test_video_path = test_video_path
        self.metadata.update({
            'render.modes': ["rgb_array"],
            'video.frames_per_second': 12
        })  # 24 frames, 2 sec

    def reset(self):
        self._step = 0
        if self._test_video_path is not None:
            # evaluate a particular video
            assert self._test_video_path in self._val_videos
            path = self._test_video_path
        elif self._test:
            # SAVI reports validation results; movi_a/b doesn't have test split
            idx = self._test_n % len(self._val_videos)
            path = self._val_videos[idx]
            self._test_n += 1
        else:
            path = np.random.choice(self._training_videos)

        try:
            with open(path, 'rb') as f:
                self._video = pickle.load(f)
        except Exception as e:
            print("Loading video %s failed" % path)
            raise RuntimeError(str(e))
        self._video_id = np.float32(
            os.path.basename(path).replace('.pkl', '').split('_')[-1])
        return self._obs()

    @property
    def episode_length(self):
        return self._video["video"].shape[0]

    def _segmentation_uint8_to_rgb(self, seg, colormap=matplotlib.cm.tab20b):
        cmap = np.array(((0, 0, 0), ) + colormap.colors)  # len=21
        ret = np.round(cmap[seg[..., 0]] * 255).astype(np.uint8)
        return ret

    def _obs(self):
        rgb = self._video["video"][self._step]  # uint8
        # background is always 0
        segmentation = self._video["segmentations"][self._step]  # uint8
        # "segmentation" will only be used for evaluation, not for training or inference
        obs = {"rgb": rgb, "segmentation": segmentation}
        return obs

    def step(self, action):
        assert self._video is not None, "Must call reset() first!"
        self._step += 1
        done = False
        if self._step == self.episode_length - 1:
            done = True
        # expose video_id for visualization purpose only
        return self._obs(), 0., done, {
            'video_id': self._video_id,
            'frame_id': self._step
        }

    def render(self, mode="human"):
        obs = self._obs()
        imgs = [
            obs['rgb'],
            self._segmentation_uint8_to_rgb(obs['segmentation'])
        ]
        frame = np.concatenate(imgs)
        if mode == "rgb_array":
            return frame
        else:
            cv2.imshow(self._dataset, frame)
            cv2.waitKey(500)

    def seed(self, seed: int = None):
        np.random.seed(seed)
