import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import defaultdict


class PaleoDataGenerator:
    """
    古气候合成数据生成器 v2.0
    用于生成已知真实因果结构、具有古气候典型统计特征的合成数据集。
    支持场景A（事件尺度）、场景B（轨道尺度）、场景C（状态转换）。
    """

    def __init__(self, scenario='A', random_seed=None):
        self.scenario = scenario
        self.rng = np.random.default_rng(random_seed)
        self.dt = None
        self.n_points = None
        self.var_names = None

        # 预设场景配置
        if scenario == 'A':
            self._init_scenario_A()
        elif scenario == 'B':
            self._init_scenario_B()
        elif scenario == 'C':
            self._init_scenario_C()
        else:
            raise ValueError(f"未知场景: {scenario}")

    # ========================
    # 场景初始化
    # ========================
    def _init_scenario_A(self):
        """场景A：事件-千年尺度，模拟0-60ka的高分辨率记录"""
        self.dt = 0.05  # ka，对应50年分辨率
        self.total_time = 65.0  # 生成65ka，截断边界效应后保留50ka
        self.n_points = int(self.total_time / self.dt)

        # 6条核心变量
        self.var_names = ['NGRIP', 'EDC_dD', 'CO2', 'CH4', 'Cariaco', 'Cheng']

        # 自回归系数（高自相关是古气候数据的核心特征）
        self.phi = {
            'NGRIP': 0.85, 'EDC_dD': 0.82, 'CO2': 0.90,
            'CH4': 0.78, 'Cariaco': 0.72, 'Cheng': 0.75
        }

        # 噪声标准差
        self.noise_std = {
            'NGRIP': 0.45, 'EDC_dD': 0.55, 'CO2': 0.80,
            'CH4': 0.70, 'Cariaco': 0.30, 'Cheng': 0.25
        }

        # 共同强迫载荷（所有变量共享的外部强迫影响强度）
        self.forcing_weights = {
            'NGRIP': 0.35, 'EDC_dD': 0.40, 'CO2': 0.25,
            'CH4': 0.20, 'Cariaco': 0.30, 'Cheng': 0.20
        }

        # 真实因果图（驱动源 -> 响应变量）及滞后 (ka)
        self.causal_edges = {
            ('NGRIP', 'EDC_dD'): 0.8,
            ('CO2', 'CH4'): 1.5,
            ('Cariaco', 'Cheng'): 2.3
        }
        # 耦合强度
        self.causal_strength = {edge: 0.40 for edge in self.causal_edges}

        # 年龄误差分布参数（1σ，单位ka）
        self.age_error_std = {
            'NGRIP': 0.65, 'EDC_dD': 0.83, 'CO2': 1.65,
            'CH4': 1.66, 'Cariaco': 0.23, 'Cheng': 0.14
        }

    def _init_scenario_B(self):
        """场景B：轨道尺度，模拟0-800ka的低分辨率记录"""
        self.dt = 1.0  # 分辨率1ka
        self.total_time = 805.0
        self.n_points = int(self.total_time / self.dt)

        self.var_names = ['d18O', 'CO2', 'CH4', 'Temp']

        # 极强自相关（冰盖惯性）
        self.phi = {'d18O': 0.97, 'CO2': 0.96, 'CH4': 0.95, 'Temp': 0.96}
        self.noise_std = {'d18O': 0.15, 'CO2': 0.12, 'CH4': 0.18, 'Temp': 0.15}

        # 共同强迫较弱（轨道强迫已被模拟为直接驱动）
        self.forcing_weights = {v: 0.05 for v in self.var_names}

        # 轨道尺度的因果图（滞后更大，信噪比更高）
        self.causal_edges = {
            ('d18O', 'CO2'): 5.0,
            ('CO2', 'CH4'): 1.5,
            ('d18O', 'Temp'): 3.0
        }
        self.causal_strength = {edge: 0.50 for edge in self.causal_edges}

        # 年龄误差小（相对于周期）
        self.age_error_std = {'d18O': 2.0, 'CO2': 4.0, 'CH4': 4.0, 'Temp': 2.0}

    def _init_scenario_C(self):
        """场景C：长期气候状态转换，模拟0-2000ka"""
        self.dt = 2.0
        self.total_time = 2010.0
        self.n_points = int(self.total_time / self.dt)

        self.var_names = ['d18O', 'd13C', 'SST']
        self.phi = {v: 0.92 for v in self.var_names}
        self.noise_std = {v: 0.18 for v in self.var_names}
        self.forcing_weights = {v: 0.10 for v in self.var_names}

        # 场景C不使用固定因果滞后，而是状态相关的相关性
        self.causal_edges = {}
        self.causal_strength = {}

        # 年龄误差较大
        self.age_error_std = {'d18O': 10.0, 'd13C': 10.0, 'SST': 10.0}

    # ========================
    # 核心生成组件
    # ========================
    def _generate_ar1(self, n, phi, noise_std):
        """生成AR(1)过程：x_t = phi*x_{t-1} + ε_t"""
        x = np.zeros(n)
        noise = self.rng.normal(0, noise_std, n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + noise[t]
        return x

    def _generate_common_forcing(self, t):
        """
        生成古气候特有的共同外部强迫
        包含：轨道趋势 + 类D/O事件的脉冲
        """
        forcing = np.zeros_like(t)

        # 1. 低频轨道趋势 (100ka周期)
        forcing += 0.5 * np.sin(2 * np.pi * t / 100.0)

        # 2. D/O式脉冲事件
        if self.scenario == 'A':
            n_do_events = 25
            do_start_times = self.rng.uniform(5, 55, n_do_events)
            for do_time in do_start_times:
                # 高斯脉冲：快速变暖，缓慢变冷（不对称）
                pulse = np.exp(-0.5 * ((t - do_time) / 0.4) ** 2)
                pulse[t < do_time] = 0  # 脉冲前为0
                # 缓慢衰减
                decay = np.exp(-(t - do_time) / 1.2)
                decay[t < do_time] = 0
                forcing += 1.5 * pulse + 0.6 * decay
        elif self.scenario == 'B':
            # 轨道尺度主要使用真实的Milankovitch周期组合
            forcing += 0.8 * np.sin(2 * np.pi * t / 41.0)  # 斜率
            forcing += 0.6 * np.sin(2 * np.pi * t / 23.0)  # 岁差
            forcing += 0.4 * np.sin(2 * np.pi * t / 100.0)  # 偏心率

        return forcing

    # ========================
    # 数据生成主函数
    # ========================
    def generate_dataset(self):
        """生成完整的合成数据集"""
        t = np.arange(0, self.total_time, self.dt)
        n = len(t)
        common_forcing = self._generate_common_forcing(t)

        # 1. 生成所有变量的AR(1)基线 + 噪声
        baselines = {}
        for var in self.var_names:
            baselines[var] = self._generate_ar1(n, self.phi[var], self.noise_std[var])

        # 2. 构建最终的信号（未施加因果滞后）
        signals = {}
        for var in self.var_names:
            signal = baselines[var].copy()
            # 加入共同强迫
            signal += self.forcing_weights[var] * common_forcing
            signals[var] = signal

        # 3. 根据因果图施加滞后影响
        #    按拓扑顺序处理，确保驱动源的影响能正确传递
        processed = set()
        # 简单的拓扑排序：找出所有驱动源
        drivers = set(edge[0] for edge in self.causal_edges)
        responders = set(edge[1] for edge in self.causal_edges)
        start_nodes = drivers - responders  # 纯粹的驱动源

        # 从纯驱动源开始，递归处理
        queue = list(start_nodes) + [v for v in self.var_names if v not in start_nodes and v not in responders]

        while queue:
            var = queue.pop(0)
            if var in processed:
                continue

            # 检查该变量的所有驱动源是否已处理
            dependencies = [d for d, r in self.causal_edges if r == var]
            if not all(d in processed for d in dependencies):
                queue.append(var)  # 放回队尾
                continue

            # 施加滞后影响
            for driver in dependencies:
                lag_ka = self.causal_edges[(driver, var)]
                lag_steps = int(lag_ka / self.dt)
                strength = self.causal_strength[(driver, var)]

                if lag_steps > 0 and lag_steps < n:
                    # 将驱动变量的过去值叠加到当前变量
                    rolled = np.zeros_like(signals[var])
                    rolled[lag_steps:] = signals[driver][:-lag_steps]
                    signals[var] += strength * rolled

            processed.add(var)

        # 处理纯响应变量（可能在队列中未出现）
        for var in self.var_names:
            if var not in processed:
                dependencies = [d for d, r in self.causal_edges if r == var]
                for driver in dependencies:
                    if driver in processed:
                        lag_ka = self.causal_edges[(driver, var)]
                        lag_steps = int(lag_ka / self.dt)
                        strength = self.causal_strength[(driver, var)]
                        if 0 < lag_steps < n:
                            rolled = np.zeros_like(signals[var])
                            rolled[lag_steps:] = signals[driver][:-lag_steps]
                            signals[var] += strength * rolled
                processed.add(var)

        # 标准化所有信号
        for var in self.var_names:
            mean = np.mean(signals[var])
            std = np.std(signals[var])
            if std > 0:
                signals[var] = (signals[var] - mean) / std

        # 截取有效区间（去除边界效应）
        if self.scenario == 'A':
            mask = (t >= 10.0) & (t <= 60.0)
        elif self.scenario == 'B':
            mask = (t >= 5.0) & (t <= 800.0)
        else:
            mask = (t >= 10.0) & (t <= 2000.0)

        t_valid = t[mask]
        signals_valid = {var: sig[mask] for var, sig in signals.items()}
        forcing_valid = common_forcing[mask]

        return t_valid, signals_valid, forcing_valid

    def inject_age_errors(self, t, signals):
        """
        为数据注入年代不确定性。
        返回一个包含多次扰动实现的数据集，供后续蒙特卡洛分析。
        """
        perturbed = {}
        for var in self.var_names:
            sigma = self.age_error_std[var]
            # 对年龄标签添加高斯噪声
            age_perturbed = t + self.rng.normal(0, sigma, len(t))
            perturbed[var] = age_perturbed
        return perturbed

    def generate_ensemble(self, n_realizations=100):
        """
        生成多个独立的随机实现。
        用于统计评估因果推断方法的稳健性（TPR, FPR）。
        """
        ensemble = []
        for i in range(n_realizations):
            t, signals, forcing = self.generate_dataset()
            # 注入年龄误差
            age_errors = self.inject_age_errors(t, signals)

            ensemble.append({
                'time': t,
                'signals': signals,
                'common_forcing': forcing,
                'age_errors': age_errors,
                'causal_edges': self.causal_edges,
                'causal_strength': self.causal_strength
            })
        return ensemble