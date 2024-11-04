import numpy as np
from numba import njit
import random

@njit
def get_sni(env, x, y, th, nbh): #if no neighbours then they're unhappy
    race = env[x,y]
    base = env[(x - nbh):(x + nbh + 1), (y - nbh):(y + nbh + 1)]
    all_nbh = (base != 0).sum() - 1
    happ = (base == race).sum() - 1
    return happ/all_nbh if all_nbh > 0 else 0

@njit
def is_not_happy(env, x, y, th, nbh): 
    sni = get_sni(env, x, y, th, nbh)
    return sni < th

class SchellingModel:
    
    def __init__(self,  sizes: tuple, th: tuple, nbh: tuple, L=100):
        self.sizes = {1: int(sizes[0]),
                      -1: int(sizes[1])}
        self.th = {1: th[0],
                   -1: th[1]}
        self.nbh = {1: int(nbh[0]),
                    -1: int(nbh[1])}
        self.L = L
        self.padding = max(self.nbh.values())
        self.gen_env()
        
    def gen_env(self):
        area = self.L**2
        env = np.full(int(area - sum(self.sizes.values())), 0.)
        env = np.append(env, np.full(self.sizes[-1], -1))
        env = np.append(env, np.full(self.sizes[1], 1))
        np.random.shuffle(env)
        env = np.resize(env, (self.L, self.L))
        full_env = np.zeros((self.L + (2 * self.padding), self.L + (2 * self.padding)))
        full_env[self.padding : -(self.padding), self.padding : -(self.padding)] = env
        self.env = full_env.copy()
        self.update_padding()

    def update_padding(self):
        self.env[:self.padding, self.padding:-(self.padding)] = self.env[-(2 * self.padding):-(self.padding), self.padding:-(self.padding)]
        self.env[-(self.padding):, self.padding:-(self.padding)] = self.env[self.padding:(2 * self.padding), self.padding:-(self.padding)]
        self.env[self.padding:-(self.padding), :self.padding] = self.env[self.padding:-(self.padding), -(2 * self.padding):-(self.padding)]
        self.env[self.padding:-(self.padding), -(self.padding):] = self.env[self.padding:-(self.padding), self.padding:(2 * self.padding)]
        self.env[:self.padding, :self.padding] = self.env[-(2 * self.padding):-(self.padding), -(2 * self.padding):-(self.padding)]
        self.env[:self.padding, -(self.padding):] = self.env[-(2 * self.padding):-(self.padding), (self.padding):(2 * self.padding)]
        self.env[-(self.padding):, :self.padding] = self.env[(self.padding):(2 * self.padding), -(2 * self.padding):-(self.padding)]
        self.env[-(self.padding):, -(self.padding):] = self.env[(self.padding):(2 * self.padding), (self.padding):(2 * self.padding)]
    
    def get_mean_happy(self):
        return sum(map(lambda x: not is_not_happy(self.env, 
                                                  x[0], 
                                                  x[1],
                                                  self.th[self.env[x[0], x[1]]], 
                                                  self.nbh[self.env[x[0], x[1]]]), self.pos))/self.pos.shape[0]
        
    def get_mean_sni(self):
        return sum(map(lambda x: get_sni(self.env,
                                         x[0],
                                         x[1],
                                         self.th[self.env[x[0], x[1]]], 
                                         self.nbh[self.env[x[0], x[1]]]), self.pos))/self.pos.shape[0]
                            
    def move(self, x: int, y: int, old_id: int):
        state = self.env.copy()[x, y]
        new_ind = int(self.empty_pos.shape[0] * random.random()) #ask if that's okay
        new_x, new_y = self.empty_pos[new_ind]
        self.env[new_x, new_y], self.env[x, y] = state, 0
        self.empty_pos[new_ind] = (x, y)
        self.pos[old_id] = [new_x, new_y]
        self.update_padding()
                            
    def one_tick(self):
        max_pos = self.pos.shape[0]
        tick = 0
        while tick < max_pos:
            x, y = self.pos[tick]
            state = self.env[x, y]
            if is_not_happy(self.env, x, y, self.th[state], self.nbh[state]):
                self.move(x, y, tick)
            tick += 1
        
    def cut_borders(self, mat):
        mat = mat[~(mat < self.padding).any(1),:]
        return mat[~(mat > (self.L + self.padding - 1)).any(1),:]
    
    def simulate(self, n=1):
        base_env = self.env.copy()
        self.sni = 0
        self.t = 0
        M = n
        while n>0:
            self.env = base_env.copy()
            self.empty_pos = self.cut_borders(np.transpose(np.where(self.env == 0)))
            self.pos = self.cut_borders(np.transpose(np.nonzero(self.env)))
            ts = 0
            happ_st = self.get_mean_happy()
            while happ_st<1:
                self.one_tick()
                happ_st = self.get_mean_happy()
                ts += 1
            self.sni += self.get_mean_sni()
            self.t += ts
            n -= 1
        self.sni /= M
        self.t /= M

class SchellingModelWithSave(SchellingModel):
    
    def __init__(self,  sizes: tuple, th: tuple, nbh: tuple, L=100):
        super().__init__(sizes, th, nbh, L)
        self.history = [self.env[1:-1].copy()]
                            
    def one_tick(self):
        super().one_tick()
        self.history.append(self.env[1:-1].copy())
# import time
# t1 = time.time()
# temp = SchellingModel((250, 250), (0.8, 0.8), (1, 1))
# temp.simulate(100)
# t2 = time.time()
# print(t2-t1)
# print(temp.t)
# print(temp.sni)
# import seaborn as sns 
# from matplotlib import pyplot as plt
# sns.heatmap(temp.history[-1])
# plt.show()
