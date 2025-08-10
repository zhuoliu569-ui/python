import numpy as np
from scipy.optimize import differential_evolution
import colour

# 因为Python模块名不能以数字开头，我们需要使用exec或importlib方式导入
import importlib.util
import sys
import os

# 动态导入2_1.py模块
spec = importlib.util.spec_from_file_location("module2_1", os.path.join(os.path.dirname(__file__), "2_1.py"))
module2_1 = importlib.util.module_from_spec(spec)
sys.modules["module2_1"] = module2_1
spec.loader.exec_module(module2_1)

# 从导入的模块中获取需要的函数
loaddata = module2_1.loaddata
create_spectral_distribution = module2_1.create_spectral_distribution
to_xyz1931 = module2_1.to_xyz1931
to_cct = module2_1.to_cct
get_rf_rg = module2_1.get_rf_rg
calculate_mel_der = module2_1.calculate_mel_der
weighted_sum_sds = module2_1.weighted_sum_sds
get_std_illuminant = module2_1.get_std_illuminant

def calculate_duv(xy):
    """计算DUV值"""
    x, y = xy[0], xy[1]
    
    # 计算u, v坐标 (CIE 1960 UCS)
    denom = -2 * x + 12 * y + 3
    if denom == 0:
        u = v = 0.0
    else:
        u = (4 * x) / denom
        v = (6 * y) / denom
    
    # 计算黑体轨迹上的点到测试点的距离
    k6 = -0.00616793
    k5 = 0.0893944
    k4 = -0.5179722
    k3 = 1.5317403
    k2 = -2.4243787
    k1 = 1.925865
    k0 = -0.471106
    
    # 计算Lfp和角度a
    Lfp = np.sqrt((u - 0.292)**2 + (v - 0.24)**2)
    
    # 避免除零错误
    if Lfp == 0:
        a = 0
    else:
        a = np.arccos((u - 0.292) / Lfp)
    
    # 计算黑体轨迹上的点到参考点的距离
    Lbb = (k6 * a**6 + k5 * a**5 + k4 * a**4 + 
           k3 * a**3 + k2 * a**2 + k1 * a + k0)
    
    # DUV是测试点到黑体轨迹的垂直距离
    Duv = Lfp - Lbb
    return Duv

# 基于2_1.py中的函数，我们需要创建一个新的优化目标

def calc_metrics_for_mel_der(W):
    """计算给定权重下的所有指标，用于mel-DER优化"""
    sds = [sd_blue, sd_green, sd_red, sd_warm_white, sd_cool_white]
    light = weighted_sum_sds(W, sds)
    xy, XYZ, u, v = to_xyz1931(light)
    T = to_cct(xy)
    Rf, Rg = get_rf_rg(light, T)
    mel_der, details = calculate_mel_der(light)
    Duv = calculate_duv(xy)
    
    return {
        'T': T, 
        'Rf': Rf, 
        'Rg': Rg, 
        'mel_der': mel_der,
        'light': light,
        'Duv': Duv
    }

def mel_der_objective(W):
    """优化目标函数：最小化mel-DER，同时满足约束条件"""
    metrics = calc_metrics_for_mel_der(W)
    
    # 检查是否为有效数值
    if not all(np.isfinite([metrics['T'], metrics['Rf'], metrics['mel_der'], metrics['Duv']])):
        return 1e6  # 返回一个大的值表示无效解
    
    # 检查约束条件
    penalty = 0
    
    # T 应该在 2500K 到 3500K 之间
    if metrics['T'] < 2500:
        penalty += (2500 - metrics['T']) * 1e3
    elif metrics['T'] > 3500:
        penalty += (metrics['T'] - 3500) * 1e3
    
    # Rf 应该在 80 到 100 之间
    if metrics['Rf'] < 80:
        penalty += (80 - metrics['Rf']) * 1e4
    elif metrics['Rf'] > 100:
        penalty += (metrics['Rf'] - 100) * 1e4
        
    # |DUV| 应该 <= 0.054
    if abs(metrics['Duv']) > 0.054:
        penalty += (abs(metrics['Duv']) - 0.054) * 1e4
        
    # 如果满足所有约束，优化目标是最小化mel-DER
    if penalty == 0:
        return metrics['mel_der']
    else:
        return 1e6 + penalty  # 添加惩罚项

def constraint_T_low_mel(W):
    """色温下限约束 (2500K)"""
    metrics = calc_metrics_for_mel_der(W)
    return metrics['T'] - 2500

def constraint_T_high_mel(W):
    """色温上限约束 (3500K)"""
    metrics = calc_metrics_for_mel_der(W)
    return 3500 - metrics['T']

def constraint_Rf_low_mel(W):
    """Rf下限约束 (80)"""
    metrics = calc_metrics_for_mel_der(W)
    return metrics['Rf'] - 80

def constraint_Rf_high_mel(W):
    """Rf上限约束 (100)"""
    metrics = calc_metrics_for_mel_der(W)
    return 100 - metrics['Rf']

def constraint_Duv(W):
    """DUV绝对值约束 (<= 0.054)"""
    metrics = calc_metrics_for_mel_der(W)
    return 0.054 - abs(metrics['Duv'])

# 可以使用差分进化算法进行优化
def optimize_for_mel_der():
    """执行针对mel-DER的优化"""
    bounds = [(0, 1)] * 5  # 5个权重，每个都在0到1之间

    # 使用差分进化算法进行优化
    result = differential_evolution(
        mel_der_objective,
        bounds=bounds,
        strategy='best1bin',
        maxiter=500,
        popsize=20,
        tol=1e-3,
        workers=-1,
        updating='deferred',
        polish=True
    )

    return result

# 加载数据并创建光谱分布对象
data_1 = loaddata("第二问数据.txt")
data_blue = np.column_stack((data_1[:, 0], data_1[:, 1]))
data_green = np.column_stack((data_1[:, 0], data_1[:, 2]))
data_red = np.column_stack((data_1[:, 0], data_1[:, 3]))
data_warm_white = np.column_stack((data_1[:, 0], data_1[:, 4]))
data_cool_white = np.column_stack((data_1[:, 0], data_1[:, 5]))
sd_blue = create_spectral_distribution(data_blue)
sd_green = create_spectral_distribution(data_green)
sd_red = create_spectral_distribution(data_red)
sd_warm_white = create_spectral_distribution(data_warm_white)
sd_cool_white = create_spectral_distribution(data_cool_white)

# 使用示例
if __name__ == "__main__":
    # 运行优化
    result = optimize_for_mel_der()
    
    if result.success:
        # 归一化权重
        best_W = result.x / np.sum(result.x)
        print("Mel-DER优化结果:")
        print("最优权重:", best_W)
        
        # 计算最终指标
        metrics = calc_metrics_for_mel_der(best_W)
        print(f"色温 T: {metrics['T']:.2f}K")
        print(f"显色指数 Rf: {metrics['Rf']:.2f}")
        print(f"色域指数 Rg: {metrics['Rg']:.2f}")
        print(f"Mel-DER: {metrics['mel_der']:.5f}")
        print(f"DUV: {metrics['Duv']:.5f}")
        
        # 验证约束条件
        print("\n约束条件验证:")
        print(f"色温是否在2500-3500K之间: {2500 <= metrics['T'] <= 3500}")
        print(f"Rf是否在80-100之间: {80 <= metrics['Rf'] <= 100}")
        print(f"|DUV|是否<=0.054: {abs(metrics['Duv']) <= 0.054}")
    else:
        print("优化失败:", result.message)