import random
import numpy as np
import fastrand, math
from ea.mod_utils import is_lnorm_key


class SSNE:
    def __init__(self, args):
        self.current_gen = 0
        self.args = args
        self.prob_reset_and_sup = args.prob_reset_and_sup

        self.frac = args.frac
        self.population_size = self.args.pop_size
        self.num_elitists = int(self.args.elite_fraction * args.pop_size)
        if self.num_elitists < 1: self.num_elitists = 1

        self.rl_policy = None
        self.selection_stats = {'elite': 0, 'selected': 0, 'discarded': 0, 'total': 0.0000001}

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[fastrand.pcg32bounded(len(offsprings))])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def regularize_weight(self, weight, mag):
        if weight > mag: weight = mag
        if weight < -mag: weight = -mag
        return weight

    def crossover_inplace(self, gene1, gene2, agent_index: int):
        # Evaluate the parents

        b_1 = None
        b_2 = None
        for param1, param2 in zip(gene1.agent.parameters(), gene2.agent.parameters()):
            # References to the variable tensors
            W1 = param1.data
            W2 = param2.data
            if len(W1.shape) == 1:
                b_1 = W1
                b_2 = W2

        for param1, param2 in zip(gene1.agent.parameters(), gene2.agent.parameters()):
            # References to the variable tensors
            W1 = param1.data
            W2 = param2.data

            if len(W1.shape) == 2:  # Weights no bias
                num_variables = W1.shape[0]
                # Crossover opertation [Indexed by row]
                num_cross_overs = fastrand.pcg32bounded(num_variables * 2)  # Lower bounded on full swaps
                for i in range(num_cross_overs):
                    receiver_choice = random.random()  # Choose which gene to receive the perturbation
                    if receiver_choice < 0.5:
                        ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                        W1[ind_cr, :] = W2[ind_cr, :]
                        b_1[ind_cr] = b_2[ind_cr]
                    else:
                        ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                        W2[ind_cr, :] = W1[ind_cr, :]
                        b_2[ind_cr] = b_1[ind_cr]

        # Evaluate the children

    def mutate_inplace(self, gene, agent_index, agent_level=False):
        trials = 5

        mut_strength = 0.1
        num_mutation_frac = 0.1
        super_mut_strength = 10
        super_mut_prob = self.prob_reset_and_sup
        reset_prob = super_mut_prob + self.prob_reset_and_sup

        num_params = len(list(gene.agent.parameters()))
        ssne_probabilities = np.random.uniform(0, 1, num_params) * 2
        model_params = gene.agent.state_dict()

        for i, key in enumerate(model_params):  # Mutate each param

            if is_lnorm_key(key):
                continue

            # References to the variable keys
            W = model_params[key]
            if len(W.shape) == 2:  # Weights, no bias
                if agent_level:
                    ssne_prob = 1.0  # ssne_probabilities[i]
                    action_prob = 1.0
                else:
                    ssne_prob = 1.0
                    action_prob = ssne_probabilities[i]

                if random.random() < ssne_prob:
                    num_variables = W.shape[0]
                    # Crossover opertation [Indexed by row]
                    for index in range(num_variables):
                        random_num_num = random.random()
                        if random_num_num <= action_prob:
                            # print(W)
                            index_list = random.sample(range(W.shape[1]), int(W.shape[1] * self.frac))
                            random_num = random.random()
                            if random_num < super_mut_prob:  # Super Mutation probability
                                for ind in index_list:
                                    W[index, ind] += random.gauss(0, super_mut_strength * W[index, ind])
                            elif random_num < reset_prob:  # Reset probability
                                for ind in index_list:
                                    W[index, ind] = random.gauss(0, 1)
                            else:  # mutation even normal
                                for ind in index_list:
                                    W[index, ind] += random.gauss(0, mut_strength * W[index, ind])

                            # Regularization hard limit
                            W[index, :] = np.clip(W[index, :].cpu(), a_min=-1000000, a_max=1000000)

    def clone(self, master, replacee, agent_index: int):  # Replace the replacee individual with master

        for target_param, source_param in zip(replacee.agent.parameters(), master.agent.parameters()):
            target_param.data.copy_(source_param.data)
        # replacee.buffer.reset()
        # replacee.buffer.add_content_of(master.buffer)

    def reset_genome(self, gene):
        for param in (gene.agent.parameters()):
            param.data.copy_(param.data)

    def pareto_dominates(self, individual1, individual2):
        better_in_one = False
        for f1, f2 in zip(individual1, individual2):
            if f1 < f2:
                return False
            elif f1 > f2:
                better_in_one = True
        return better_in_one

    def non_dominated_sorting(self, population):
        pareto_fronts = []
        domination_counts = {i: 0 for i in range(len(population))}
        dominated_solutions = {i: [] for i in range(len(population))}

        current_front = []

        for i in range(len(population)):
            for j in range(len(population)):
                if self.pareto_dominates(population[i], population[j]):
                    dominated_solutions[i].append(j)
                elif self.pareto_dominates(population[j], population[i]):
                    domination_counts[i] += 1

            if domination_counts[i] == 0:
                current_front.append(i)

        pareto_fronts.append(current_front)

        while len(current_front) > 0:
            next_front = []
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            pareto_fronts.append(next_front)
            current_front = next_front

        return pareto_fronts

    def fitness_split(self, fitness):
        """
        将单一的奖励值分解为多个目标
        :param reward: 总奖励值
        :param delta_enemy: 对敌方造成的伤害
        :param delta_deaths: 击杀敌方单位
        :param delta_ally: 己方单位受到的伤害
        :return: 分解后的多个目标值
        """
        reward = fitness[0]
        delta_enemy = fitness[1]
        delta_deaths = fitness[2]
        delta_ally = fitness[3]
        # 战斗效能：击杀敌方单位和造成的伤害
        combat_effectiveness = 0.5 * (delta_enemy + delta_deaths)

        # 生存能力：己方单位受到的伤害（负向指标，越低越好）
        survivability = 0.3 * -delta_ally  # 负数表示惩罚

        # 团队贡献：综合战斗效能和生存能力的影响
        team_contribution = 0.2 * reward  # 总奖励值作为团队贡献

        return combat_effectiveness, survivability, team_contribution

    def calculate_crowding_distance(self, front, population):
        distances = {i: 0 for i in front}
        num_objectives = len(population[0])

        for m in range(num_objectives):
            sorted_front = sorted(front, key=lambda i: population[i][m])

            # # 给排序列表中的第一个和最后一个个体赋值无穷大，以防止选择时被过早丢弃
            # distances[sorted_front[0]] = distances[sorted_front[-1]] = float('inf')

            # 计算内部个体的拥挤距离（即相邻个体的目标值差异）
            for i in range(1, len(sorted_front) - 1):
                distances[sorted_front[i]] += (population[sorted_front[i + 1]][m] - population[sorted_front[i - 1]][m])

        return distances

    def epoch(self, pop, fitness_evals, agent_index, agent_level=False):
        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)

        if self.args.Pareto:
            population = [self.fitness_split(fitness) for fitness in fitness_evals]

            pareto_fronts = self.non_dominated_sorting(population)

            # Calculate crowding distance for each front
            crowding_distances = {}
            for front in pareto_fronts:
                crowding_distances.update(self.calculate_crowding_distance(front, population))

            # Selection step: select individuals from Pareto fronts, using crowding distance for tie-breaking
            index_rank = []
            for front in pareto_fronts:
                # sorted_front = sorted(front, key=lambda i: crowding_distances[i], reverse=True)
                # selected.extend(sorted_front[:len(pop) // len(pareto_fronts)])  # Select individuals from each front
                index_rank.extend(front)
            # Elitism: Keep the best individuals
            elitist_index = index_rank[:self.num_elitists]
        else:
            index_rank = np.argsort(fitness_evals)[::-1]

            elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)

        # Figure out unselected candidates
        unselects = [];
        new_elitists = []
        for i in range(self.population_size):
            if i not in offsprings and i not in elitist_index:
                unselects.append(i)
        random.shuffle(unselects)

        # COMPUTE RL_SELECTION RATE
        if self.rl_policy is not None:  # RL Transfer happened
            self.selection_stats['total'] += 1.0

            if self.rl_policy in elitist_index:
                self.selection_stats['elite'] += 1.0
            elif self.rl_policy in offsprings:
                self.selection_stats['selected'] += 1.0
            elif self.rl_policy in unselects:
                self.selection_stats['discarded'] += 1.0
            self.rl_policy = None

        # Elitism step, assigning elite candidates to some unselects
        for i in elitist_index:
            try:
                replacee = unselects.pop(0)
            except:
                replacee = offsprings.pop(0)
            new_elitists.append(replacee)
            self.clone(master=pop[i], replacee=pop[replacee], agent_index=agent_index)

        # Crossover between elite and offsprings for the unselected genes with 100 percent probability

        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[fastrand.pcg32bounded(len(unselects))])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists)
            off_j = random.choice(offsprings)
            self.clone(master=pop[off_i], replacee=pop[i], agent_index=agent_index)
            self.clone(master=pop[off_j], replacee=pop[j], agent_index=agent_index)

            if agent_level:
                if random.random() < 0.5:
                    self.clone(master=pop[i], replacee=pop[j], agent_index=agent_index)
                else:
                    self.clone(master=pop[j], replacee=pop[i], agent_index=agent_index)
            else:
                self.crossover_inplace(pop[i], pop[j], agent_index)

        # Mutate all genes in the population except the new elitists
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.args.mutation_prob:
                    self.mutate_inplace(pop[i], agent_index=agent_index, agent_level=agent_level)

        return new_elitists[0]


def unsqueeze(array, axis=1):
    if axis == 0:
        return np.reshape(array, (1, len(array)))
    elif axis == 1:
        return np.reshape(array, (len(array), 1))
