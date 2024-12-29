import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import random
from collections import deque
import itertools
import DQN
import PPO
from env_cfg import Config, TestDemand, Agent
import torch
from torch.utils.tensorboard import SummaryWriter


def get_init_len(init):
    """
    Calculate total number of elements in a 1D array or list of lists.
    :type init: iterable, list or (list of lists)
    :rtype: int
    """
    is_init_array = all([isinstance(x, (float, int, np.int64)) for x in init])
    if is_init_array:
        init_len = len(init)
    else:
        init_len = len(list(itertools.chain.from_iterable(init)))
    return init_len



class BeerGame(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, n_agents=4, n_turns_per_game=100,test_mode=False):
        super().__init__()
        c = Config()
        config, unparsed = c.get_config()
        self.config = config
        self.test_mode = test_mode
        if self.test_mode:
            self.test_demand_pool = TestDemand()

        self.curGame = 1 # The number associated with the current game (counter of the game)
        self.curTime = 0
        self.m = 10             #window size
        self.totIterPlayed = 0  # total iterations of the game, played so far in this and previous games
        self.players = self.createAgent()  # create the agents
        self.T = 0
        self.demand = []
        self.orders = []
        self.shipments = []
        self.rewards = []
        self.cur_demand = 0

        self.ifOptimalSolExist = self.config.ifOptimalSolExist
        self.getOptimalSol()

        self.totRew = 0    # it is reward of all players obtained for the current player.
        #self.totalReward = 0
        self.n_agents = n_agents

        self.n_turns = n_turns_per_game
        seed  = random.randint(0,1000000)
        self.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.totalTotal = 0

        # Agent 0 has 5 (-2, ..., 2) + AO
        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(5),gym.spaces.Discrete(5),gym.spaces.Discrete(5),gym.spaces.Discrete(5)]))

        # Seemingly useless code deleted

        # Define the observation space, x holds the size of each part of the state
        # x = [750, 750, 170, 45, 45]
        # oob = []
        # for _ in range(self.m):
        #   for ii in range(len(x)):
        #     oob.append(x[ii])
        # self.observation_space = gym.spaces.Tuple(tuple([spaces.MultiDiscrete(oob)] * 4))
        #
        # print("Observation space:")
        # print(self.observation_space)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def createAgent(self):
      agentTypes = self.config.agentTypes
      return [Agent(i,self.config.ILInit[i], self.config.AOInit, self.config.ASInit[i],
                                self.config.c_h[i], self.config.c_p[i], self.config.eta[i],
                                agentTypes[i],self.config) for i in range(self.config.NoAgent)]


    def resetGame(self, demand, ):
        self.demand = demand
        self.playType='test'
        self.curTime = 0
        self.curGame += 1
        self.totIterPlayed += self.T
        self.T = self.planHorizon()
        #self.totalReward = 0

        self.deques = []
        # observable state initialize
        for i in range(self.n_agents):
            deques = {}
            deques[f'current_stock_minus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'current_stock_plus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'OO'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AS'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AO'] = deque([0] * self.m, maxlen=self.m)
            self.deques.append(deques)

        # reset the required information of player for each episode

        for k in range(0,self.config.NoAgent):
            self.players[k].resetPlayer(self.T)

        # update OO when there are initial IL,AO,AS
        self.update_OO()


    def reset(self):
        if self.test_mode:
            demand = self.test_demand_pool.next()
            if not self.test_demand_pool:           #if run out of testing data
                self.test_demand_pool = TestDemand()
        else:
            demand = [random.randint(0,2) for _ in range(1002)]

        self.resetGame(demand)
        # observations = [None] * self.n_agents

        self.deques = []
        for i in range(self.n_agents):
            deques = {}
            deques[f'current_stock_minus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'current_stock_plus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'OO'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AS'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AO'] = deque([0] * self.m, maxlen=self.m)
            self.deques.append(deques)

        # prepend current observation
        # get current observation, prepend to deque
        for i in range(self.n_agents):
            curState = self.players[i].getCurState(self.curTime)
            self.deques[i]['current_stock_minus'].appendleft(int(curState[0]))
            self.deques[i]['current_stock_plus'].appendleft(int(curState[1]))
            self.deques[i]['OO'].appendleft(int(curState[2]))
            self.deques[i]['AS'].appendleft(int(curState[3]))
            self.deques[i]['AO'].appendleft(int(curState[4]))

        # return entire m observations
        obs = [[], [], [], []]
        for i in range(self.n_agents):
            spaces = {}
            for j in range(self.m):
                obs[i].append(self.deques[i]['current_stock_minus'][j])
                obs[i].append(self.deques[i]['current_stock_plus'][j])
                obs[i].append(self.deques[i]['OO'][j])
                obs[i].append(self.deques[i]['AS'][j])
                obs[i].append(self.deques[i]['AO'][j])
                # spaces[f'current_stock_minus{j}'] = self.deques[i]['current_stock_minus'][j]
                # spaces[f'current_stock_plus{j}'] = self.deques[i]['current_stock_plus'][j]
                # spaces[f'OO{j}'] = self.deques[i]['OO'][j]
                # spaces[f'AS{j}'] = self.deques[i]['AS'][j]
                # spaces[f'AO{j}'] = self.deques[i]['AO'][j]

            # observations[i] = spaces

        obs_array = np.array([np.array(row) for row in obs])
        return obs_array  # observations #self._get_observations()


    def step(self, action:list):
        if get_init_len(action) != self.n_agents:
            raise error.InvalidAction(f'Length of action array must be same as n_agents({self.n_agents})')
        if any(np.array(action) < 0):
            raise error.InvalidAction(f"You can't order negative amount. You agents actions are: {action}")

        self.handleAction(action)
        self.next()

        self.orders = action

        for i in range(self.n_agents):
            self.players[i].getReward()
        self.rewards = [1 * self.players[i].curReward for i in range(0, self.config.NoAgent)]

        if self.curTime == self.T+1:
            self.done = [True] * 4
        else:
            self.done = [False] * 4


        # get current observation, prepend to deque
        for i in range(self.n_agents):
            curState = self.players[i].getCurState(self.curTime)
            self.deques[i]['current_stock_minus'].appendleft(int(curState[0]))
            self.deques[i]['current_stock_plus'].appendleft(int(curState[1]))
            self.deques[i]['OO'].appendleft(int(curState[2]))
            self.deques[i]['AS'].appendleft(int(curState[3]))
            self.deques[i]['AO'].appendleft(int(curState[4]))

        # return entire m observations
        obs = [[],[],[],[]]
        observations = [None] * self.n_agents
        for i in range(self.n_agents):
            spaces = {}
            for j in range(self.m):
              obs[i].append(self.deques[i]['current_stock_minus'][j])
              obs[i].append(self.deques[i]['current_stock_plus'][j])
              obs[i].append(self.deques[i]['OO'][j])
              obs[i].append(self.deques[i]['AS'][j])
              obs[i].append(self.deques[i]['AO'][j])

        obs_array = np.array([np.array(row) for row in obs])
        state = obs_array #observations #self._get_observations()
        return state, self.rewards, self.done, {}



    def handleAction(self, action):
        # get random lead time
        #leadTime = random.randint(self.config.leadRecOrderLow[0], self.config.leadRecOrderUp[0])
        self.cur_demand = self.demand[self.curTime]
        # set AO
        BS = False
        self.players[0].AO[self.curTime] += self.demand[self.curTime]       #orders from customer, add directly to the retailer arriving order
        for k in range(0, self.config.NoAgent):
            if k >= 0:  #recording action
                self.players[k].action = np.zeros(5)        #one-hot transformation
                self.players[k].action[action[k]] = 1
                BS = False
            else:
                raise NotImplementedError
                #self.getAction(k)
                #BS = True

            # updates OO and AO at time t+1
            self.players[k].OO += self.players[k].actionValue(self.curTime, self.playType, BS = BS)     #open order level update
            leadTime = random.randint(self.config.leadRecOrderLow[k], self.config.leadRecOrderUp[k])        #order
            if self.players[k].agentNum < self.config.NoAgent-1:

                if k>=0:
                    self.players[k + 1].AO[self.curTime + leadTime] += self.players[k].actionValue(self.curTime,
                                                                                                   self.playType,
                                                                                                   BS=False)  # TODO(yan): k+1 arrived order contains my own order and the order i received from k-1
                else:
                    raise NotImplementedError
                    #self.players[k + 1].AO[self.curTime + leadTime] += self.players[k].actionValue(self.curTime,
                    #                                                                               self.playType,
                    #                                                                               BS=True)  # open order level update

    def next(self):
        # get a random leadtime for shipment
        leadTimeIn = random.randint(self.config.leadRecItemLow[self.config.NoAgent - 1],
                                    self.config.leadRecItemUp[self.config.NoAgent - 1])

        # handle the most upstream recieved shipment
        self.players[self.config.NoAgent-1].AS[self.curTime + leadTimeIn] += self.players[self.config.NoAgent-1].actionValue(self.curTime, self.playType, BS=True)
                                                                #the manufacture gets its ordered beer after leadtime

        self.shipments = []
        for k in range(self.config.NoAgent-1,-1,-1): # [3,2,1,0]

            # get current IL and Backorder
            current_IL = max(0, self.players[k].IL)
            current_backorder = max(0, -self.players[k].IL)

            # increase IL and decrease OO based on the action, for the next period
            self.players[k].recieveItems(self.curTime)

            # observe the reward
            possible_shipment = min(current_IL + self.players[k].AS[self.curTime],
                                    current_backorder + self.players[k].AO[self.curTime])       #if positive IL, ship all beer or all they needs, if backorders, ship all k-1 needs
            self.shipments.append(possible_shipment)

            # plan arrivals of the items to the downstream agent
            if self.players[k].agentNum > 0:
                leadTimeIn = random.randint(self.config.leadRecItemLow[k-1], self.config.leadRecItemUp[k-1])
                self.players[k-1].AS[self.curTime + leadTimeIn] += possible_shipment

            # update IL
            self.players[k].IL -= self.players[k].AO[self.curTime]

            # observe the reward
            self.players[k].getReward()
            #rewards = [-1 * self.players[i].curReward for i in range(0, self.config.NoAgent)]

            # update next observation
            self.players[k].nextObservation = self.players[k].getCurState(self.curTime + 1)

        if self.config.ifUseTotalReward:  # default is false
            # correction on cost at time T
            if self.curTime == self.T:
                self.getTotRew()

        self.curTime += 1

    def getAction(self, k):
        self.players[k].action = np.zeros(self.config.actionListLenOpt)

        if self.config.demandDistribution == 2:
            if self.curTime and self.config.use_initial_BS <= 4:
                self.players[k].action[np.argmin(np.abs(np.array(self.config.actionListOpt) - \
                                                        max(0, (self.players[k].int_bslBaseStock - (
                                                                    self.players[k].IL + self.players[k].OO -
                                                                    self.players[k].AO[self.curTime])))))] = 1
            else:
                self.players[k].action[np.argmin(np.abs(np.array(self.config.actionListOpt) - \
                                                        max(0, (self.players[k].bsBaseStock - (
                                                                    self.players[k].IL + self.players[k].OO -
                                                                    self.players[k].AO[self.curTime])))))] = 1
        else:
            self.players[k].action[np.argmin(np.abs(np.array(self.config.actionListOpt) - \
                                                    max(0, (self.players[k].bsBaseStock - (
                                                                self.players[k].IL + self.players[k].OO -
                                                                self.players[k].AO[self.curTime])))))] = 1

    def getTotRew(self):
      totRew = 0
      for i in range(self.config.NoAgent):
        # sum all rewards for the agents and make correction
        totRew += self.players[i].cumReward

      for i in range(self.config.NoAgent):
        self.players[i].curReward += self.players[i].eta*(totRew - self.players[i].cumReward) #/(self.T)


    def planHorizon(self):
      # TLow: minimum number for the planning horizon # TUp: maximum number for the planning horizon
      #output: The planning horizon which is chosen randomly.
      return random.randint(self.n_turns, self.n_turns)# self.config.TLow,self.config.TUp)

    def update_OO(self):
        for k in range(0,self.config.NoAgent):
            if k < self.config.NoAgent - 1:
                self.players[k].OO = sum(self.players[k+1].AO) + sum(self.players[k].AS)
            else:
                self.players[k].OO = sum(self.players[k].AS)

    def getOptimalSol(self):
        # if self.config.NoAgent !=1:
        if self.config.NoAgent != 1 and 1 == 2:
            # check the Shang and Song (2003) condition.
            for k in range(self.config.NoAgent - 1):
                if not (self.players[k].c_h == self.players[k + 1].c_h and self.players[k + 1].c_p == 0):
                    self.ifOptimalSolExist = False

            # if the Shang and Song (2003) condition satisfied, it runs the algorithm
            if self.ifOptimalSolExist == True:
                calculations = np.zeros((7, self.config.NoAgent))
                for k in range(self.config.NoAgent):
                    # DL_high
                    calculations[0][k] = ((self.config.leadRecItemLow + self.config.leadRecItemUp + 2) / 2 \
                                          + (self.config.leadRecOrderLow + self.config.leadRecOrderUp + 2) / 2) * \
                                         (self.config.demandUp - self.config.demandLow - 1)
                    if k > 0:
                        calculations[0][k] += calculations[0][k - 1]
                    # probability_high
                    nominator_ch = 0
                    low_denominator_ch = 0
                    for j in range(k, self.config.NoAgent):
                        if j < self.config.NoAgent - 1:
                            nominator_ch += self.players[j + 1].c_h
                        low_denominator_ch += self.players[j].c_h
                    if k == 0:
                        high_denominator_ch = low_denominator_ch
                    calculations[2][k] = (self.players[0].c_p + nominator_ch) / (
                                self.players[0].c_p + low_denominator_ch + 0.0)
                    # probability_low
                    calculations[3][k] = (self.players[0].c_p + nominator_ch) / (
                                self.players[0].c_p + high_denominator_ch + 0.0)
                # S_high
                calculations[4] = np.round(np.multiply(calculations[0], calculations[2]))
                # S_low
                calculations[5] = np.round(np.multiply(calculations[0], calculations[3]))
                # S_avg
                calculations[6] = np.round(np.mean(calculations[4:6], axis=0))
                # S', set the base stock values into each agent.
                for k in range(self.config.NoAgent):
                    if k == 0:
                        self.players[k].bsBaseStock = calculations[6][k]

                    else:
                        self.players[k].bsBaseStock = calculations[6][k] - calculations[6][k - 1]
                        if self.players[k].bsBaseStock < 0:
                            self.players[k].bsBaseStock = 0
        elif self.config.NoAgent == 1:
            if self.config.demandDistribution == 0:
                self.players[0].bsBaseStock = np.ceil(
                    self.config.c_h[0] / (self.config.c_h[0] + self.config.c_p[0] + 0.0)) * ((
                                                                                                         self.config.demandUp - self.config.demandLow - 1) / 2) * self.config.leadRecItemUp
        elif 1 == 1:
            f = self.config.f
            f_init = self.config.f_init
            for k in range(self.config.NoAgent):
                self.players[k].bsBaseStock = f[k]
                self.players[k].int_bslBaseStock = f_init[k]

    def render(self, mode='human'):
        # if mode != 'human':
        #     raise NotImplementedError(f'Render mode {mode} is not implemented yet')
        # print("")
        print('\n' + '=' * 20)
        print('Turn:     ', self.curTime)
        stocks = [p.IL for p in self.players]
        print('Stocks:   ', ", ".join([str(x) for x in stocks]))
        print('Orders:   ', self.orders)
        print('Shipments:', self.shipments)
        print('Rewards:', self.rewards)
        print('Customer demand: ', self.cur_demand)

        AO = [p.AO[self.curTime] for p in self.players]
        AS = [p.AS[self.curTime] for p in self.players]

        print('Arrived Order: ', AO)
        print('Arrived Shipment: ', AS)

        OO = [p.OO for p in self.players]
        print('Working Order: ', OO)


        # print('Last incoming orders:  ', self.next_incoming_orders)
        # print('Cum holding cost:  ', self.cum_stockout_cost)
        # print('Cum stockout cost: ', self.cum_holding_cost)
        # print('Last holding cost: ', self.holding_cost)
        # print('Last stockout cost:', self.stockout_cost)

def use_dqn():
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 100
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    state_dim = 200
    action_dim = 5
    agent = DQN.DQN_net(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
    epoch = 100# 打开一个文件，用于写入奖励数据
    with open("rewards_rn_rn_demand_new.txt", "w") as file:



    # log_dir = "logs/test1"
    # summary_writer = SummaryWriter(log_dir)
        for i in range(epoch):
            rewards = []
            # env_name = 'CartPole-v0'
            # env = gym.make(env_name)
            random.seed(i)
            np.random.seed(i)
            # env.seed(0)
            torch.manual_seed(i)
            replay_buffer = DQN.ReplayBuffer(buffer_size)
            # state_dim = env.observation_space.shape[0]
            # action_dim = env.action_space.n

            env = BeerGame()
            obs = env.reset()
            #print(obs)
            #env.render()
            done = False


            turn = 0
            while not done:
                turn+=1
                # print(env.action_space)
                ac_list = []
                action_3 = DQN.get_q_action(agent, obs)

                env.players[3].action = action_3

                for k in range(4):
                    env.getAction(k)
                    #print(np.argmax(env.players[k].action))
                    ac_list.append(min(4,np.argmax(env.players[k].action)))

                rnd_action = list(env.action_space.sample())
                # # rnd_action = [3,1,2,0]
                # rnd_action[3] = action_3
                #ac_list.append(action_3)
                next_obs, reward, done_list, _ = env.step(rnd_action)

                rewards.append(-np.mean(reward))



                DQN.update(obs, next_obs, action_3, reward, False, replay_buffer, agent)
                # next_obs, reward, done_list, _ = env.step(rnd_action)
                # print(next_obs)

                obs = next_obs
                done = all(done_list)
                # if i%10==0:
                #     rewards_list = [env.players[3].IL, env.players[3].OO, rnd_action[3], reward[3]]
                #     rewards_str = ','.join(map(str, rewards_list)) + '\n'
                #     file.write(rewards_str)
                #env.render()
            #print(np.sum(rewards))
            file.write(str(np.sum(rewards))+ "\n")




            #summary_writer.add_scalar("cost/BS+BS", np.sum(rewards), i)

        #summary_writer.close()



def use_ppo():
    actor_lr = 1e-4
    critic_lr = 1e-4
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    epoch = 100
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")


    torch.manual_seed(0)
    state_dim = 200
    action_dim = 5
    agent = PPO.PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)

    with open("rewards_bs_ppo.txt", "w") as file:
        for i in range(epoch):
            rewards = []
            env = BeerGame()
            obs = env.reset()
            #print(obs)
            #env.render()
            done = False
            turn = 1
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            while not done:

                turn+=1
                # print(env.action_space)
                ac_list = []
                #print(obs)
                action_3 = agent.take_action(obs)

                env.players[3].action = action_3

                for k in range(3):
                    env.getAction(k)
                    #print(np.argmax(env.players[k].action))
                    ac_list.append(min(4,np.argmax(env.players[k].action)))
                ac_list.append(action_3)
                rnd_action = list(env.action_space.sample())
                # rnd_action = [3,1,2,0]
                rnd_action[3] = action_3
                next_obs, reward, done_list, _ = env.step(ac_list)
                rewards.append(-np.mean(reward))
                #print(obs)
                transition_dict['states'].append(np.array(obs).reshape(-1,1).squeeze(1))
                transition_dict['actions'].append(action_3)
                transition_dict['next_states'].append(np.array(next_obs).reshape(-1,1).squeeze(1))
                transition_dict['rewards'].append(sum(reward))
                transition_dict['dones'].append(0)
                obs = next_obs
                if turn%10==0:
                    agent.update(transition_dict)
                    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}


                # next_obs, reward, done_list, _ = env.step(rnd_action)
                # print(next_obs)
                done = all(done_list)
                #env.render()

            #transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            #print(np.sum(rewards))
            file.write(str(np.sum(rewards)) + "\n")
def draw():
    with open("rewards_bs_ppo.txt", "r") as file:

        rewards = file.readlines()
    rewards_bs_ppo_list = [float(reward.strip()) for reward in rewards]

    # with open("rewards_rn_rn_demand_new.txt", "r") as file:
    #
    #     rewards = file.readlines()
    #
    # rewards_rd_rd_list = [float(reward.strip()) for reward in rewards]
    # with open("rewards_bs_bs_demand_new.txt", "r") as file:
    #
    #     rewards = file.readlines()
    #
    # rewards_bs_bs_list = [float(reward.strip()) for reward in rewards]
    with open("rewards_bs_dqn.txt", "r") as file:

        rewards = file.readlines()

    rewards_bs_dqn_list = [float(reward.strip()) for reward in rewards]
    log_dir = "logs/test3"
    summary_writer = SummaryWriter(log_dir)
    turn = 0
    for i,j in zip(rewards_bs_dqn_list, rewards_bs_ppo_list):
        turn+=1
        #print(i,j)
        summary_writer.add_scalars("cost2", {'BS_DQN': i,'BS_PPO': j}, turn)

    summary_writer.close()


def draw_2():

    with open("actions_rn_rn_demand_new.txt", "r") as file:

        rewards = file.readlines()


    rewards_rd_rd_list = [list(map(float, line.strip().split(','))) for line in rewards]
    with open("actions_bs_bs_demand_new.txt", "r") as file:

        rewards = file.readlines()

    rewards_bs_bs_list = [list(map(float, line.strip().split(','))) for line in rewards]
    with open("actions_bs_dqn_demand_new.txt", "r") as file:

        rewards = file.readlines()

    rewards_bs_dqn_list = [list(map(float, line.strip().split(','))) for line in rewards]
    log_dir = "logs/test2"
    summary_writer = SummaryWriter(log_dir)
    turn = 0
    for i,j,k in zip(rewards_rd_rd_list, rewards_bs_bs_list, rewards_bs_dqn_list):
        #print(i,j,k)
        il_i = i[0]
        oo_i = i[1]
        ac_i = i[2]
        r_i = i[3]
        il_j = j[0]
        oo_j= j[1]
        ac_j = j[2]
        r_j = j[3]
        il_k = k[0]
        oo_k = k[1]
        ac_k = k[2]
        r_k = k[3]
        turn+=1
        #print(i,j,k)
        summary_writer.add_scalars("il", {'RD_RD': il_i,'BS_BS': il_j,'BS_DQN': il_k}, turn)
        summary_writer.add_scalars("oo", {'RD_RD': oo_i, 'BS_BS': oo_j, 'BS_DQN': oo_k}, turn)
        summary_writer.add_scalars("ac", {'RD_RD': ac_i, 'BS_BS': ac_j, 'BS_DQN': ac_k}, turn)
        summary_writer.add_scalars("r", {'RD_RD': r_i, 'BS_BS': r_j, 'BS_DQN': r_k}, turn)

    summary_writer.close()


if __name__ == "__main__":
    use_dqn()
    draw()
    #use_ppo()
    #draw_2()



    print(1)