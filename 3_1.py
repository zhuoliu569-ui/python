import numpy as np
from scipy.optimize import differential_evolution
import colour
import importlib.util
import sys
import os

# 动态导入2_1.py模块
spec = importlib.util.spec_from_file_location("module2_1", os.path.join(os.path.dirname(__file__), "2_1.py"))
module2_1 = importlib.util.module_from_spec(spec)
sys.modules["module2_1"] = module2_1
spec.loader.exec_module(module2_1)

# 获取2_1.py中的函数
loaddata = module2_1.loaddata
create_spectral_distribution = module2_1.create_spectral_distribution
to_xyz1931 = module2_1.to_xyz1931
to_cct = module2_1.to_cct
get_rf_rg = module2_1.get_rf_rg
calculate_mel_der = module2_1.calculate_mel_der
weighted_sum_sds = module2_1.weighted_sum_sds

# 全局变量用于传递给优化函数
global_target_sd = None
global_component_sds = None

def load_third_question_data(filename="第三问数据.txt"):
    """加载第三问数据"""
    data = np.loadtxt(filename)
    wavelengths = data[:, 0]
    spectra = data[:, 1:]  # 所有光谱列
    return wavelengths, spectra

def create_target_sds(wavelengths, spectrum):
    """为单个目标光谱创建光谱分布对象"""
    spectral_data = np.column_stack((wavelengths, spectrum))
    return create_spectral_distribution(spectral_data)

def calc_spectral_similarity(W, target_sd, component_sds):
    """计算光谱相似度"""
    # 合成光谱
    synthesized_sd = weighted_sum_sds(W, component_sds)
    
    # 计算均方误差
    mse = 0
    count = 0
    for wavelength in target_sd.wavelengths:
        if wavelength in synthesized_sd.wavelengths:
            diff = target_sd[wavelength] - synthesized_sd[wavelength]
            mse += diff ** 2
            count += 1
    
    if count > 0:
        mse /= count
    else:
        mse = float('inf')
    
    return mse, synthesized_sd

def spectral_matching_objective(W):
    """光谱匹配优化目标函数"""
    global global_target_sd, global_component_sds
    mse, synthesized_sd = calc_spectral_similarity(W, global_target_sd, global_component_sds)
    
    # 检查是否为有效数值
    if not np.isfinite(mse):
        return 1e6  # 返回一个大的值表示无效解
    
    return mse

def optimize_spectral_matching(target_sd, component_sds):
    """执行光谱匹配优化"""
    global global_target_sd, global_component_sds
    global_target_sd = target_sd
    global_component_sds = component_sds
    
    bounds = [(0, 1)] * len(component_sds)  # 每个权重都在0到1之间

    # 使用差分进化算法进行优化
    result = differential_evolution(
        spectral_matching_objective,
        bounds=bounds,
        strategy='best1bin',
        maxiter=300,
        popsize=15,
        tol=1e-4,
        workers=1,  # 使用单线程避免pickle问题
        updating='immediate',
        polish=True,
        seed=42
    )

    return result

def simulate_third_question_spectra():
    """模拟第三问光谱数据的主函数"""
    # 加载第二问的组件光谱数据
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
    
    component_sds = [sd_blue, sd_green, sd_red, sd_warm_white, sd_cool_white]
    
    # 加载第三问的目标光谱数据
    wavelengths, spectra = load_third_question_data("第三问数据.txt")
    
    # 对每个目标光谱进行优化
    results = []
    for i in range(min(5, spectra.shape[1])):  # 仅演示前5个光谱
        print(f"正在优化第 {i+1} 个目标光谱...")
        
        # 创建目标光谱分布对象
        target_spectrum = spectra[:, i]
        target_sd = create_target_sds(wavelengths, target_spectrum)
        
        # 运行优化
        result = optimize_spectral_matching(target_sd, component_sds)
        
        if result.success:
            # 归一化权重
            best_W = result.x / np.sum(result.x)
            
            # 计算最终指标
            mse, synthesized_sd = calc_spectral_similarity(best_W, target_sd, component_sds)
            
            # 计算色温等相关参数
            xy, XYZ, u, v = to_xyz1931(synthesized_sd)
            T = to_cct(xy)
            
            results.append({
                'target_index': i,
                'weights': best_W,
                'mse': mse,
                'T': T,
                'success': True
            })
            
            print(f"第 {i+1} 个光谱优化成功:")
            print(f"  最优权重: {best_W}")
            print(f"  光谱均方误差: {mse:.6f}")
            print(f"  色温: {T:.2f}K")
        else:
            results.append({
                'target_index': i,
                'weights': None,
                'mse': None,
                'T': None,
                'success': False,
                'message': result.message
            })
            print(f"第 {i+1} 个光谱优化失败: {result.message}")
    
    return results

# 使用示例
if __name__ == "__main__":
    print("开始模拟第三问光谱数据...")
    results = simulate_third_question_spectra()
    
    # 输出总结
    successful_fits = sum(1 for r in results if r['success'])
    total_targets = len(results)
    
    print(f"\n优化完成总结:")
    print(f"成功拟合光谱数量: {successful_fits}/{total_targets}")
    if successful_fits > 0:
        avg_mse = np.mean([r['mse'] for r in results if r['success']])
        print(f"平均光谱均方误差: {avg_mse:.6f}")