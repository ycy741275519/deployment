# -*- coding: utf-8 -*-
# @File  : Sensor.py
# @Author: ycy
# @Date  : 2019/12/3
# @Desc  :传感器
import numpy as np
class Sensor(object):
    def __init__(self,id,pos):
        self.id=id                  #监测区域编号
        self.pos=pos                #传感器坐标 (第一列x轴，第二列y轴)

    # 生成传感器
    # num 传感器数量
    # L   正方形区域长度
    @classmethod
    def generateSensor(cls, num, L):
        sensors = []
        for i in range(num):
            sensors.append(Sensor(i, [np.random.random() * L, np.random.random() * L]))
        return sensors
    def get_X(self):
        return self.pos[0]

    def get_Y(self):
        return self.pos[1]

    def set_pos(self,pos):
        self.pos = pos

    def get_pos(self):
        return self.pos

