import gym
import numpy as np
from copo.algo_copo.constants import *
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import get_activation_fn, try_import_tf, try_import_torch
from ray.rllib.utils.typing import TensorType, ModelConfigDict
import torch.nn as nn


torch, nn = try_import_torch()


class NeiValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        if config.get("use_gae"):
            if self.config[USE_CENTRALIZED_CRITIC]:
                # TODO cc is not implemented
                raise ValueError()

            else:
                def nei_value(ob, prev_action, prev_reward, *state):
                    model_out, _ = self.model(
                        {
                            SampleBatch.CUR_OBS: convert_to_torch_tensor(
                                np.asarray([ob]), self.device),
                            SampleBatch.PREV_ACTIONS: convert_to_torch_tensor(
                                np.asarray([prev_action]), self.device),
                            SampleBatch.PREV_REWARDS: convert_to_torch_tensor(
                                np.asarray([prev_reward]), self.device),
                            "is_training": False,
                        }, [
                            convert_to_torch_tensor(np.asarray([s]), self.device)
                            for s in state
                        ], convert_to_torch_tensor(np.asarray([1]), self.device)
                    )

                    return self.model.get_nei_value()[0]

                def global_value(ob, prev_action, prev_reward, *state):
                    model_out, _ = self.model(
                        {
                            SampleBatch.CUR_OBS: convert_to_torch_tensor(
                                np.asarray([ob]), self.device),
                            SampleBatch.PREV_ACTIONS: convert_to_torch_tensor(
                                np.asarray([prev_action]), self.device),
                            SampleBatch.PREV_REWARDS: convert_to_torch_tensor(
                                np.asarray([prev_reward]), self.device),
                            "is_training": False,
                        }, [
                            convert_to_torch_tensor(np.asarray([s]), self.device)
                            for s in state
                        ], convert_to_torch_tensor(np.asarray([1]), self.device)
                    )
                    return self.model.get_global_value()[0]

                self.get_nei_value = nei_value
                self.get_global_value = global_value

        else:
            raise ValueError()


def register_copo_torch_model():
    ModelCatalog.register_custom_model("copo_model", CoPOTorchModel)


class CoPOTorchModel(TorchModelV2, nn.Module):
    def __init__(
        self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
        model_config: ModelConfigDict, name: str
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        activation = model_config.get("fcnet_activation")
        hiddens = model_config.get("fcnet_hiddens", [])
        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")  # False
        free_log_std = model_config.get("free_log_std")

        use_centralized_critic = model_config[USE_CENTRALIZED_CRITIC]
        self.use_centralized_critic = use_centralized_critic

        # Generate free-floating bias variables for the second half of
        # the outputs.
        if free_log_std:
            raise ValueError()

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size, out_size=size,
                    initializer=normc_initializer(1.0), activation_fn=activation
                )
            )
            prev_layer_size = size

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            raise ValueError()
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size, out_size=hiddens[-1],
                        initializer=normc_initializer(1.0), activation_fn=activation
                    )
                )
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = SlimFC(
                    in_size=prev_layer_size, out_size=num_outputs,
                    initializer=normc_initializer(0.01), activation_fn=None
                    )
            else:
                raise ValueError()

        self._hidden_layers = nn.Sequential(*layers)

        # ===== Build original value function =====
        self._value_branch_separate = None
        if not vf_share_layers:  # True
            prev_vf_layer_size = int(np.product(obs_space.shape))
            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size, out_size=size,
                        initializer=normc_initializer(1.0), activation_fn=activation
                    )
                )
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)
        else:
            raise ValueError()

        self._value_branch = SlimFC(
            in_size=prev_layer_size, out_size=1,
            initializer=normc_initializer(1.0), activation_fn=None
        )

        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

        # ===== Build neighbours value function =====
        prev_vf_nei_layer_size = int(np.product(obs_space.shape))
        vf_nei_layers = []
        for size in hiddens:
            vf_nei_layers.append(
                SlimFC(
                    in_size=prev_vf_nei_layer_size, out_size=size,
                    initializer=normc_initializer(1.0), activation_fn=activation
                )
            )
            prev_vf_nei_layer_size = size
        self._value_nei_branch_separate = nn.Sequential(*vf_nei_layers)

        self._value_nei_branch = SlimFC(
            in_size=prev_vf_nei_layer_size, out_size=1,
            initializer=normc_initializer(1.0), activation_fn=None
        )

        # ===== Build global value function =====
        prev_vf_global_layer_size = int(np.product(obs_space.shape))
        vf_global_layers = []
        for size in hiddens:
            vf_global_layers.append(
                SlimFC(
                    in_size=prev_vf_global_layer_size, out_size=size,
                    initializer=normc_initializer(1.0), activation_fn=activation
                )
            )
            prev_vf_global_layer_size = size
        self._value_global_branch_separate = nn.Sequential(*vf_global_layers)

        self._value_global_branch = SlimFC(
            in_size=prev_vf_global_layer_size, out_size=1,
            initializer=normc_initializer(1.0), activation_fn=None
        )

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        logits = self._logits(self._hidden_layers(self._last_flat_in))

        self._value_out = self._value_branch(
            self._value_branch_separate(self._last_flat_in)
        ).squeeze(1)

        self._nei_value_out = self._value_nei_branch(
            self._value_nei_branch_separate(self._last_flat_in)
        ).squeeze(1)

        self._global_value_out = self._value_global_branch(
            self._value_global_branch_separate(self._last_flat_in)
        ).squeeze(1)

        return logits, state

    def value_function(self, cc_obs=None) -> TensorType:
        assert self._last_flat_in is not None, "must call forward() first"
        return self._value_out

    def get_nei_value(self, cc_obs=None):
        return self._nei_value_out

    def get_global_value(self, cc_obs=None):
        return self._global_value_out
