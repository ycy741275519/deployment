#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from .Sensor import  Sensor
class FAIndividual:

    '''
    individual of firefly algorithm
    萤火虫类
    '''

    def __init__(self,  vardim, bound):
        '''
        vardim: dimension of variables 空间维度
        bound: boundaries of variables 移动范围限制
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.                                                               #适应值
        self.R=2.5                                                                    #充电范围
        self.a = 4.32*1e-4                             #充电模型系数
        self.b = 0.2316
        self.c1=-3                                       #目标函数系数
        self.c2=2
        self.uWorst=self.a/((self.R+self.b)**2)        #最差的充电效率
    def generate(self):
        '''
        generate a random chromsome for firefly algorithm
        随机生成位置
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        for i in range(0, len):
            self.chrom[i] = self.bound[0, i] + \
                (self.bound[1, i] - self.bound[0, i]) * rnd[i]

    def calculateFitness(self,sensors):
        '''
        calculate the fitness of the chromsome
        计算适应值
        '''
        #计算充电范围内传感器个数
        self.num=0
        self.maxU=0                                     #最大的最差充电效率与节点充电效率比，为当前位置最差的充电效率比
        for sensor in sensors:
            d=np.linalg.norm(self.chrom-np.array(sensor.get_pos()))
            if(d<=self.R):
                self.num+=1
                u=self.a/((d+self.b)**2)                       #该节点的充电效率
                uRate=self.uWorst/u                            #最差充电效率与当前充电效率之比
                if uRate>self.maxU:
                    self.maxU = uRate
        #计算适应值
        if self.num==0:
            self.fitness=np.float('inf')
        else:
            self.fitness=self.c1*self.num+self.c2*self.maxU

    def plotChangeRange(self,sensors,L):
        fig = plt.figure(figsize=(4, 4))
        x = []
        y = []
        for sensor in sensors:
            x.append(sensor.get_X())
            y.append(sensor.get_Y())
        plt.scatter(x, y)
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = self.chrom[0] + self.R * np.cos(theta)
        y = self.chrom[1] + self.R * np.sin(theta)
        plt.plot(x,y)
        plt.xticks(np.arange(L+1))
        plt.yticks(np.arange(L+1))
        plt.xlim(0,L)
        plt.ylim(0,L)
        plt.show()



if __name__ == '__main__':
    bound = np.tile([[0], [10]], 25)
    sensors =Sensor.generateSensor(30, 10)
    ind = FAIndividual(2,bound)
    ind.generate()
    print(ind.chrom)
    # ind.calculateFitness()
    ind.plotChangeRange(sensors,10)

