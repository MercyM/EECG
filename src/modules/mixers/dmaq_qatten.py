import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
from .dmaq_qatten_weight import Qatten_Weight
from .dmaq_si_weight import DMAQ_SI_Weight
from .dmaq_simple_weight import DMAQ_Simple_Weight
from .GraphMix.GNNs.gnn import GNN


class DMAQ_QattenMixer(nn.Module):
    def __init__(self, args):
        super(DMAQ_QattenMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1

        self.attention_weight = Qatten_Weight(args)
        self.si_weight = DMAQ_SI_Weight(args) if self.args.is_adv_attention else DMAQ_Simple_Weight(args)

        if args.graphmixer:
            # mixing GNN
            hypernet_embed = self.args.hypernet_embed
            self.embed_dim = args.mixing_embed_dim
            self.rnn_hidden_dim = args.rnn_hidden_dim
            combine_type = 'gin'
            self.mixing_GNN = GNN(num_input_features=1, hidden_layers=[self.embed_dim],
                                  state_dim=self.state_dim, hypernet_embed=hypernet_embed,
                                  weights_operation='abs',
                                  combine_type=combine_type)

            # attention mechanism
            self.enc_obs = True
            obs_dim = self.rnn_hidden_dim
            if self.enc_obs:
                self.obs_enc_dim = 16
                self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, self.obs_enc_dim),
                                                 nn.ReLU())
                self.obs_dim_effective = self.obs_enc_dim
            else:
                self.obs_encoder = nn.Sequential()
                self.obs_dim_effective = obs_dim

            self.W_attn_query = nn.Linear(self.obs_dim_effective, self.obs_dim_effective, bias=False)
            self.W_attn_key = nn.Linear(self.obs_dim_effective, self.obs_dim_effective, bias=False)

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = th.sum(agent_qs, dim=-1)
        return v_tot

    def calc_adv(self, agent_qs, states, actions, max_q_i):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        adv_w_final = self.si_weight(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)

        if self.args.is_minus_one:
            adv_tot = th.sum(adv_q * (adv_w_final - 1.), dim=1)
        else:
            adv_tot = th.sum(adv_q * adv_w_final, dim=1)
        return adv_tot

    def calc(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, states, actions, max_q_i)
            return adv_tot

    def forward(self, agent_qs, states, actions=None, max_q_i=None, is_v=False,
                agent_obs=None,
                team_rewards=None,
                hidden_states=None, EA=False):
        bs = agent_qs.size(0)

        w_final, v, attend_mag_regs, head_entropies = self.attention_weight(agent_qs, states, actions)
        w_final = w_final.view(-1, self.n_agents) + 1e-10
        v = v.view(-1, 1).repeat(1, self.n_agents)
        v /= self.n_agents

        agent_qs = agent_qs.view(-1, self.n_agents)
        agent_qs = w_final * agent_qs + v
        if not is_v:
            max_q_i = max_q_i.view(-1, self.n_agents)
            max_q_i = w_final * max_q_i + v

        y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        if self.args.graphmixer: #and EA == False:
            bs2 = states.size(0)
            states = states.reshape(-1, self.state_dim)
            agent_qs = agent_qs.view(-1, self.n_agents, 1)

            # find the agents which are alive
            alive_agents = 1. * (th.sum(agent_obs, dim=3) > 0).view(-1, self.n_agents)

            # create a mask for isolating nodes which are dead by taking the outer product of the above tensor with itself
            alive_agents_temp1 = alive_agents.unsqueeze(2)
            alive_agents_temp2 = alive_agents.unsqueeze(1)

            alive_agents_tensor = th.zeros_like(alive_agents_temp1, dtype=th.float32)
            alive_agents_tensor[alive_agents_temp1 == True] = 1

            alive_agents_tensor2 = th.zeros_like(alive_agents_temp2, dtype=th.float32)
            alive_agents_tensor2[alive_agents_temp2 == True] = 1

            alive_agents_mask = th.bmm(alive_agents_tensor, alive_agents_tensor2)

            # encode hidden states
            encoded_hidden_states = self.obs_encoder(hidden_states)
            encoded_hidden_states = encoded_hidden_states.contiguous().view(-1, self.n_agents, self.obs_dim_effective)

            # adjacency based on the attention mechanism
            attn_query = self.W_attn_query(encoded_hidden_states)
            attn_key = self.W_attn_key(encoded_hidden_states)
            attn = th.matmul(attn_query, th.transpose(attn_key, 1, 2)) / np.sqrt(self.obs_dim_effective)

            batch_adj = attn * alive_agents_mask  # completely isolate the dead agents in the graph

            GNN_inputs = agent_qs
            local_reward_fractions, y_graph = self.mixing_GNN(GNN_inputs, batch_adj, states, self.n_agents)

            # state-dependent bias
            # v = self.V(states).view(-1, 1, 1)
            q_tot = v_tot + y_graph.view(bs2, -1, 1)

            # effective local rewards
            if team_rewards is None:
                local_rewards = None
            else:
                local_rewards = local_reward_fractions.view(bs2, -1, self.n_agents) * team_rewards.repeat(1, 1,
                                                                                                          self.n_agents)
            return q_tot, attend_mag_regs, head_entropies, local_rewards, alive_agents.view(bs2, -1, self.n_agents)
        else:
            return v_tot, attend_mag_regs, head_entropies
