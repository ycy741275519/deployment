import numpy as np
from FAG.FAIndividual import  FAIndividual
import random
import copy
import matplotlib.pyplot as plt
from FAG.Sensor import Sensor

class FireflyAlgorithm:
    '''
    The class for firefly algorithm
    '''


    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        '''
        sizepop: population sizepop        总数
        vardim: dimension of variables     空间维度
        bound: boundaries of variables     移动范围限制
        MAXGEN: termination condition       最大迭代次数
        param: algorithm required parameters, it is a list which is consisting of [beta0, gamma, alpha]
        '''
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.population = []                                  #萤火虫实例列表
        self.fitness = np.zeros((self.sizepop, 1))            #适应值列表
        self.trace = np.zeros((self.MAXGEN, 2))               #适应值追踪
        self.params = params

    def initialize(self):
        '''
        initialize the population
        生成萤火虫
        '''
        for i in range(0, self.sizepop):
            ind = FAIndividual(self.vardim, self.bound)            #创建萤火虫
            ind.generate()                                                      #随机生成位置
            self.population.append(ind)                                         #将萤火虫加入实例列表

    def displayPos(self,sensors,L,gen):
        """显示充电范围"""
        fig = plt.figure(figsize=(4, 4))
        x = []
        y = []
        for sensor in sensors:
            x.append(sensor.get_X())
            y.append(sensor.get_Y())
        plt.scatter(x, y)
        for p in self.population:
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = p.chrom[0] + p.R * np.cos(theta)
            y = p.chrom[1] + p.R * np.sin(theta)
            plt.plot(x, y)
        plt.xticks(np.arange(L + 1))
        plt.yticks(np.arange(L + 1))
        plt.xlim(0, L)
        plt.ylim(0, L)
        plt.savefig("E:\\code\\deployment\\FAG\\result\\fa-"+str(gen)+'.png')


    def evaluate(self,sensors):
        '''
        evaluation of the population fitnesses
        计算适应值，更新适应值列表
        '''
        for i in range(0, self.sizepop):
            self.population[i].calculateFitness(sensors)
            self.fitness[i] = self.population[i].fitness

    def solve(self,sensors,L):
        '''
        evolution process of firefly algorithm
        '''
        self.t = 0                                                      #迭代次数初始为0
        self.initialize()                                               #萤火虫初始化
        self.displayPos(sensors,L,self.t)
        self.evaluate(sensors)                                                 #计算适应值
        best = np.min(self.fitness)                                     #获取最大适应值
        bestIndex = np.argmin(self.fitness)                             #获取适应值列表最大适应值索引
        self.best = copy.deepcopy(self.population[bestIndex])           #获取最佳萤火虫实例
        self.avefitness = np.mean(self.fitness)
        self.trace[self.t, 0] = self.best.fitness       #追踪信息
        self.trace[self.t, 1] =  self.avefitness
        print("Generation %d: optimal function value is: %f; average function value is %f" % (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        while (self.t < self.MAXGEN - 1):
            self.t += 1
            self.move(sensors)
            self.displayPos(sensors, L,self.t)
            self.evaluate(sensors)
            best = np.min(self.fitness)
            bestIndex = np.argmin(self.fitness)
            if best < self.best.fitness:#更新最优点
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            self.trace[self.t, 0] = self.best.fitness
            self.trace[self.t, 1] = self.avefitness
            print("Generation %d: optimal function value is: %f; average function value is %f" % (
                self.t, self.trace[self.t, 0], self.trace[self.t, 1]))

        print("Optimal function value is: %f; " %
              self.trace[self.t, 0])
        print("Optimal solution is:")
        print(self.best.chrom)
        # self.printResult()
        #########################
        print(self.best.fitness)

    def move(self,sensors):
        '''
        move the a firefly to another brighter firefly
        '''
        for i in range(0, self.sizepop):
            for j in range(0, self.sizepop):
                if self.fitness[j] < self.fitness[i]:
                    r = np.linalg.norm(
                        self.population[i].chrom - self.population[j].chrom)                            #计算萤火虫距离
                    beta = self.params[0] * \
                           np.exp(-1 * self.params[1] * (r ** 2))                                         #计算吸引力
                    self.population[i].chrom += beta * (self.population[j].chrom - self.population[
                        i].chrom) + self.params[2] * np.random.uniform(low=-1, high=1, size=self.vardim)   #移动
                    for k in range(0, self.vardim):                                                       #移动范围检查
                        if self.population[i].chrom[k] < self.bound[0, k]:
                            self.population[i].chrom[k] = self.bound[0, k]
                        if self.population[i].chrom[k] > self.bound[1, k]:
                            self.population[i].chrom[k] = self.bound[1, k]
                    self.population[i].calculateFitness(sensors)                                                  #更新适应值
                    self.fitness[i] = self.population[i].fitness

    def printResult(self):
        '''
        plot the result of the firefly algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Firefly Algorithm for function optimization")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    sensors = Sensor.generateSensor(30, 10)
    bound = np.tile([[0], [10]], 2)
    fa = FireflyAlgorithm(20, 2, bound, 20, [1.0, 0.9, 0.3])
    fa.solve(sensors,10)
