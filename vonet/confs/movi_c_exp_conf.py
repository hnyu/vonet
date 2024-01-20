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

from alf.config_util import pre_config

pre_config({
    "_CONFIG._USER.dataset": "movi_c",
    "_CONFIG._USER.batch_size": 32,
    "_CONFIG._USER.beta": 20,
    "_CONFIG._USER.slot_size": 128,
})

from vonet.confs import movi_vonet_conf
