# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: ycy
# @Date  : 2019/12/18
# @Desc  :主程序
import numpy as np
import matplotlib.pyplot as plt
from FAG.FA import FireflyAlgorithm
from FAG.Sensor import Sensor
from sko.PSO import PSO

R = 2.5                  #充电范围
A = 4.32 * 1e-4  # 充电模型系数
B = 0.2316
uWorst=A/ ((R + B) ** 2)
c1=-3                      #控制个数
c2=2                       #控制效率
def calculateFitness(x,sensors):
    '''
    calculate the fitness of the chromsome
    计算适应值
    '''
    # 计算充电范围内传感器个数
    num = 0
    maxU = 0  # 最大的最差充电效率与节点充电效率比，为当前位置最差的充电效率比
    for sensor in sensors:
        d = np.linalg.norm(x - np.array(sensor.get_pos()))
        if (d <= R):
            num += 1
            u = A/ ((d + B) ** 2)  # 该节点的充电效率
            uRate = uWorst / u  # 最差充电效率与当前充电效率之比
            if uRate > maxU:
                maxU = uRate
    # 计算适应值
    if num == 0:
        fitness = np.float('inf')
    else:
         fitness = c1 * num + c2 * maxU
    return fitness



if __name__ == '__main__':
    # 生成传感器
    sensors = Sensor.generateSensor(30, 10)
    # 萤火虫算法
    bound = np.tile([[0], [10]], 2)
    fa = FireflyAlgorithm(20, 2, bound, 20, [1.0, 0.9, 0.3])
    fa.solve(sensors, 10)



    pso = PSO(func=calculateFitness, dim=2, sensors=sensors,pop=20, max_iter=20, lb=[0,0], ub=[10,10], w=0.8, c1=0.5, c2=0.5)
    pso.run(sensors)
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    # plt.plot(pso.gbest_y_hist)
