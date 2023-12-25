import pandas as pd
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# EVChargingDataGenerator类: 生成EV充电数据
class EVChargingDataGenerator:
    def __init__(self):
        # EV（电动汽车）正态分布参数
        self.MU_1TC = 18; self.SIGMA_1TC = 3.3   # 家庭充电模式开始时间的平均值和标准差
        self.MU_1TDIS = 8; self.SIGMA_1TDIS = 3.24  # 家庭充电模式结束时间的平均值和标准差
        self.MU_2TC = 8.5; self.SIGMA_2TC = 3.3  # 公共充电模式开始时间的平均值和标准差
        self.MU_2TDIS = 17.5; self.SIGMA_2TDIS = 3.24  # 公共充电模式结束时间的平均值和标准差

        # 时间和电池相关参数
        self.DELTA_T = 0.25  # 时间间隔（四分之一个小时）
        self.SOC_CON_A = 0.1; self.SOC_CON_B = 0.3  # EV初始电池状态(SOC)的最小和最大值
        self.SOC_MIN_A = 0.4; self.SOC_MIN_B = 0.6  # EV最小充电需求的最小和最大值
        self.SOC_MAX_A = 0.8; self.SOC_MAX_B = 1.0  # EV最大充电容量的最小和最大值

    def getHomeEV(self, n):
        # 使用正态分布生成充电开始和结束时间
        t_c = np.random.randn(n) * self.SIGMA_1TC + self.MU_1TC
        t_c = np.where((self.MU_1TC - 12 < t_c) & (t_c <= 24), t_c, t_c - 24)

        t_dis = np.random.randn(n) * self.SIGMA_1TDIS + self.MU_1TDIS
        t_dis = np.where((0 < t_dis) & (t_dis <= self.MU_1TDIS + 12), t_dis, t_dis + 24)

        # 转换时间为时隙索引
        J_c = np.ceil(t_c / self.DELTA_T).astype(int)
        J_c[J_c == 0] = 96
        J_dis = np.floor(t_dis / self.DELTA_T).astype(int)
        J_dis[J_dis == 0] = 96

        # 生成电池状态的随机值
        SOC_con = np.random.uniform(self.SOC_CON_A, self.SOC_CON_B, n)
        SOC_min = np.random.uniform(self.SOC_MIN_A, self.SOC_MIN_B, n)
        SOC_max = np.random.uniform(self.SOC_MAX_A, self.SOC_MAX_B, n)

        # 创建包含所有数据的 DataFrame
        EV_data = pd.DataFrame({'t_c': t_c, 't_dis': t_dis, 'J_c': J_c, 'J_dis': J_dis,
                                'SOC_con': SOC_con, 'SOC_min': SOC_min, 'SOC_max': SOC_max})
        return EV_data

    def getPublicEV(self, n):
        # 使用正态分布生成充电开始和结束时间
        t_c = np.random.randn(n) * self.SIGMA_2TC + self.MU_2TC
        t_c = np.where((0 < t_c) & (t_c <= self.MU_2TC + 12), t_c, t_c - 24)

        t_dis = np.random.randn(n) * self.SIGMA_2TDIS + self.MU_2TDIS
        t_dis = np.where((self.MU_2TDIS - 12 < t_dis) & (t_dis <= 24), t_dis, t_dis + 24)

        # 转换时间为时隙索引
        J_c = np.ceil(t_c / self.DELTA_T).astype(int)
        J_c[J_c == 0] = 96
        J_dis = np.floor(t_dis / self.DELTA_T).astype(int)
        J_dis[J_dis == 0] = 96

        # 生成电池状态的随机值
        SOC_con = np.random.uniform(self.SOC_CON_A, self.SOC_CON_B, n)
        SOC_min = np.random.uniform(self.SOC_MIN_A, self.SOC_MIN_B, n)
        SOC_max = np.random.uniform(self.SOC_MAX_A, self.SOC_MAX_B, n)

        # 创建包含所有数据的 DataFrame
        EV_data = pd.DataFrame({'t_c': t_c, 't_dis': t_dis, 'J_c': J_c, 'J_dis': J_dis,
                                'SOC_con': SOC_con, 'SOC_min': SOC_min, 'SOC_max': SOC_max})
        return EV_data


class EVChargingOptimizer:
    def __init__(self, P_BASIC, CAP_BAT_EV=30, ETA_EV=0.9, P_SLOW_EV=3.5, P_FAST_EV=10, DELTA_T=0.25):
        # 初始化参数
        self.P_BASIC = P_BASIC  # 基本负载数据
        self.CAP_BAT_EV = CAP_BAT_EV  # 电池容量（单位：KW/H）
        self.ETA_EV = ETA_EV  # 充电效率
        self.P_SLOW_EV = P_SLOW_EV  # 慢速充电功率
        self.P_FAST_EV = P_FAST_EV  # 快速充电功率
        self.DELTA_T = DELTA_T  # 时间间隔
        self.solver = cp.GLPK_MI  # 定义求解器

    def homeChargingPattern(self, EV):
        # 家庭充电模式下的优化模型
        n = len(EV)
        P_basic_home = np.roll(self.P_BASIC, -48)  # 将时间调整到以12点为起点

        # 创建二进制优化变量
        x = cp.Variable((n, 96), boolean=True)

        # 定义约束和目标函数
        constraints, objective = self._defineOptimization(EV, x, P_basic_home, mode='home')

        # 求解优化问题
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)

        # 返回优化结果
        return x.value, P_basic_home, (P_basic_home + cp.sum(cp.multiply(x, self.P_SLOW_EV), axis=0)).value

    def publicChargingPattern(self, EV):
        # 公共充电模式下的优化模型
        n = len(EV)
        P_basic_public = self.P_BASIC.copy()  # 保持原始负载数据不变

        # 创建二进制优化变量
        x = cp.Variable((n, 96), boolean=True)

        # 定义约束和目标函数
        constraints, objective = self._defineOptimization(EV, x, P_basic_public, mode='public')

        # 求解优化问题
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)

        # 返回优化结果
        return x.value, P_basic_public, (P_basic_public + cp.sum(cp.multiply(x, self.P_SLOW_EV), axis=0)).value

    def _defineOptimization(self, EV, x, P_basic, mode='home'):
        # 定义通用优化模型
        n = len(EV)
        constraints = []

        # 车辆未接入时的时隙应为0
        for i in range(n):
            constraints += [x[i, j] == 0 for j in range(96) if j < EV['J_c'][i] or j >= EV['J_dis'][i]]

        # 有紧急充电需求的EV，整行为定值
        for i in range(n):
            if EV['CUI'][i] < 0:
                J_end = min(EV['J_dis'][i], int(EV['J_c'][i] + np.floor(
                    ((EV['SOC_max'][i] - EV['SOC_con'][i]) * self.CAP_BAT_EV) / (self.P_FAST_EV * self.ETA_EV * self.DELTA_T))))
                constraints += [x[i, j] == 1 for j in range(int(EV['J_c'][i]), J_end)]
                constraints += [x[i, j] == 0 for j in range(J_end, int(EV['J_dis'][i]))]

        # 定义总负载
        P_total_SOC_crd = P_basic + cp.sum(cp.multiply(x, self.P_SLOW_EV), axis=0)

        # 定义目标函数：最小化负载高峰和低谷之间的差异
        objective = cp.Minimize(cp.max(P_total_SOC_crd) - cp.min(P_total_SOC_crd))

        return constraints, objective


# EVDataVisualizer类: 可视化EV数据和充电模式
class EVDataVisualizer:
    def __init__(self):
        pass

    def printEVData(self, EV, title):
        # EV数据可视化
        plt.figure(figsize=(15, 5))

        # EV到达时隙的频数直方图
        plt.subplot(1, 3, 1)
        plt.hist(EV['J_c'], bins=96, range=(0.5, 96.5))
        plt.title('EV到达时间槽直方图 - ' + title)
        plt.xlabel('到达时间槽')
        plt.ylabel('频率')

        # EV到达时间的频率直方图
        plt.subplot(1, 3, 2)
        plt.hist(EV['t_c'], bins=24, range=(0, 24), density=True)
        plt.title('EV到达时间频率直方图 - ' + title)
        plt.xlabel('到达时间 (小时)')
        plt.ylabel('频率')

        # EV电池状态SOC的散点图
        plt.subplot(1, 3, 3)
        plt.scatter(np.arange(len(EV)), EV['SOC_con'], label='SOC_con')
        plt.scatter(np.arange(len(EV)), EV['SOC_min'], label='SOC_min')
        plt.scatter(np.arange(len(EV)), EV['SOC_max'], label='SOC_max')
        plt.title('EV电池状态SOC散点图 - ' + title)
        plt.xlabel('EV编号')
        plt.ylabel('SOC')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def figureResult(self, P_basic, P_SOC_crd, title):
        # 充电模式结果可视化
        plt.figure(figsize=(12, 6))
        plt.plot(P_basic, label='基础负载')
        plt.plot(P_SOC_crd, label='协调后负载', linestyle='--')
        plt.xlabel('时间槽')
        plt.ylabel('负载 (KW)')
        plt.title('充电模式 - ' + title)
        plt.legend()
        plt.show()

def main():
    # 实例化类
    def main():
        # 实例化数据生成器、优化器和可视化器
        P_BASIC = pd.read_csv('basicLoadDatacsv.csv', header=None).values.squeeze()
        print(P_BASIC.shape)
        data_generator = EVChargingDataGenerator()
        optimizer = EVChargingOptimizer(P_BASIC)
        visualizer = EVDataVisualizer()

        # 生成家庭和公共充电模式下的EV数据
        EV_home = data_generator.getHomeEV(1000)
        EV_public = data_generator.getPublicEV(1000)

        # 家庭充电模式下的优化
        x_home, P_basic_home, P_SOC_crd_home = optimizer.homeChargingPattern(EV_home)

        # 公共充电模式下的优化
        x_public, P_basic_public, P_SOC_crd_public = optimizer.publicChargingPattern(EV_public)

        # 可视化家庭充电模式下的EV数据
        visualizer.printEVData(EV_home, "家庭充电模式")

        # 可视化公共充电模式下的EV数据
        visualizer.printEVData(EV_public, "公共充电模式")

        # 可视化家庭充电模式的优化结果
        visualizer.figureResult(P_basic_home, P_SOC_crd_home, "家庭充电模式优化结果")

        # 可视化公共充电模式的优化结果
        visualizer.figureResult(P_basic_public, P_SOC_crd_public, "公共充电模式优化结果")


if __name__ == "__main__":
    main()
