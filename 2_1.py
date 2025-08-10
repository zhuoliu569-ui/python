import colour
import numpy as np
from scipy.optimize import minimize, differential_evolution
import numba

# 物理常数
h = 6.62607015e-34
c = 299792458
K = 1.380649e-23
c1 = 2 * h * np.pi * c ** 2
c2 = c * h / K

def loaddata(filename="data.txt"):
    """加载光谱数据"""
    return np.loadtxt(filename)

def create_spectral_distribution(spectral_data):
    """创建光谱分布对象"""
    # spectral_data: shape (N, 2)
    return colour.SpectralDistribution(dict(zip(spectral_data[:, 0], spectral_data[:, 1])))

def to_xyz1931(sd):
    """计算CIE 1931 XYZ及色度坐标和uv坐标"""
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    XYZ = colour.sd_to_XYZ(sd, cmfs)
    xy = colour.XYZ_to_xy(XYZ)
    denom = XYZ[0] + 15 * XYZ[1] + 3 * XYZ[2]
    if denom == 0:
        u = v = 0.0
    else:
        u = 4 * XYZ[0] / denom
        v = 6 * XYZ[1] / denom
    return xy, XYZ, u, v

def to_xyz1964(data):
    """计算CIE 1964 XYZ"""
    wavelength_arr, energy_arr = data[:, 0], data[:, 1]
    spectral_data = dict(zip(wavelength_arr, energy_arr))
    sd = colour.SpectralDistribution(spectral_data)
    cmfs = colour.MSDS_CMFS["CIE 1964 10 Degree Standard Observer"]
    return colour.sd_to_XYZ(sd, cmfs)

def to_cct(xy):
    """色温计算"""
    n = (xy[0] - 0.3320) / (xy[1] - 0.1858)
    return -437 * n ** 3 + 3601 * n ** 2 - 6861 * n + 5514.31

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
def blackbody(lambda_nm, T):
    """黑体辐射公式"""
    lam = np.asarray(lambda_nm, dtype=float) * 1e-9
    return c1 * lam ** (-5) / (np.exp(c2 / (lam * T)) - 1)

def blackbody_uv(T):
    """黑体uv坐标"""
    wavelengths = np.arange(360, 830, dtype=float)
    intensities = blackbody(wavelengths, T)
    spectral_data = dict(zip(wavelengths, intensities))
    sd = colour.SpectralDistribution(spectral_data)
    _, _, u, v = to_xyz1931(sd)
    return u, v

def L(Lambda_nm, T):
    """黑体归一化光谱 (公式2)"""
    Lambda_nm = np.asarray(Lambda_nm, dtype=float)
    exponent = 1.4388e7 / (Lambda_nm * T)
    return Lambda_nm ** (-5) / (np.exp(exponent) - 1)


def get_std_illuminant(T):
    """标准光源归一化光谱
    根据VCIRS 01-2025计算参照照明体
    T: 相关色温(K)
    """
    wavelengths = np.arange(380, 781, dtype=float)
    
    if T <= 5000:
        # 黑体辐射体 (公式1)
        values = L(wavelengths, T) / L(560, T)
        return np.column_stack((wavelengths, values))
    else:
        # 日光照明体 (公式3)
        # 计算色品坐标x,y
        if T <= 7000:
            # 公式5
            x = -4.6070e9 / T**3 + 2.9678e6 / T**2 + 0.09911e3 / T + 0.244063
            y = -2.0064e9 / T**3 + 1.9018e6 / T**2 + 0.24748e3 / T + 0.237040
        else:
            # 公式6
            x = -2.0064e9 / T**3 + 1.9018e6 / T**2 + 0.24748e3 / T + 0.237040
            y = -3.0000e9 / T**3 + 2.8723e6 / T**2 + 0.14675e3 / T + 0.224820
        
        # 计算中间参数M1, M2 (公式4)
        denominator = 0.0241 + 0.2562 * x - 0.7341 * y
        M1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / denominator
        M2 = (0.0300 - 3.1442 * x + 3.6015 * y) / denominator
        
        # 加载S0(λ), S1(λ), S2(λ)光谱数据
        # 数据来源于表B.1: 用于计算日光照明体SPD的基函数
        s_data = np.array([
            [300, 0.04, 0.28, 0.00], [305, 0.02, 0.26, 0.00], [310, 6.00, 4.50, 2.00],
            [315, 17.60, 13.45, 3.00], [320, 29.80, 22.40, 4.00], [325, 42.45, 32.20, 6.25],
            [330, 55.30, 42.00, 8.50], [335, 56.50, 41.30, 8.15], [340, 57.30, 40.60, 7.80],
            [345, 59.55, 41.10, 7.25], [350, 61.80, 41.60, 6.70], [355, 61.65, 39.80, 6.00],
            [360, 61.50, 38.00, 5.30], [365, 61.15, 40.20, 5.70], [370, 68.80, 42.40, 6.10],
            [375, 66.10, 40.45, 4.55], [380, 63.40, 38.50, 3.00], [385, 64.60, 38.75, 2.10],
            [390, 65.80, 39.00, 1.20], [395, 80.30, 35.20, 0.05], [400, 94.80, 43.40, -1.10],
            [405, 99.80, 46.85, -0.80], [410, 104.90, 44.30, -0.50], [415, 105.35, 45.10, -0.60],
            [420, 105.95, 43.90, -0.70], [425, 105.30, 40.50, -0.95], [430, 96.80, 37.10, -1.20],
            [435, 105.35, 36.90, -1.90], [440, 113.90, 38.70, -2.60], [445, 119.75, 38.30, -2.75],
            [450, 125.50, 35.25, -2.90], [455, 125.65, 34.90, -2.85], [460, 125.50, 32.60, -2.80],
            [465, 123.40, 30.25, -2.70], [470, 121.30, 27.90, -2.60], [475, 121.30, 26.10, -2.60],
            [480, 121.30, 22.30, -2.60], [485, 117.40, 24.20, -2.60], [490, 113.50, 20.10, -1.80],
            [495, 113.30, 18.15, -1.65], [500, 113.10, 16.20, -1.50], [505, 111.95, 14.70, -1.40],
            [510, 110.80, 13.09, -1.30], [515, 108.65, 12.30, -1.25], [520, 106.50, 8.60, -1.20],
            [525, 107.80, 7.35, -1.10], [530, 108.65, 6.10, -1.00], [535, 107.05, 5.15, -0.75],
            [540, 105.30, 4.20, -0.50], [545, 104.35, 3.05, -0.40], [550, 104.40, 1.90, -0.30],
            [555, 102.20, 0.95, -0.15], [560, 100.00, 0.00, 0.00], [565, 98.00, -0.80, 0.10],
            [570, 96.55, -1.60, 0.20], [575, 95.00, -2.55, 0.35], [580, 95.10, -3.50, 0.50],
            [585, 92.10, -3.50, 1.30], [590, 89.10, -3.50, 2.10], [595, 88.80, -4.65, 2.65],
            [600, 90.50, -5.80, 3.20], [605, 90.40, -6.80, 3.65], [610, 90.30, -7.20, 4.10],
            [615, 88.45, -7.90, 4.40], [620, 89.30, -8.60, 4.70], [625, 86.20, -9.05, 4.90],
            [630, 84.50, -9.50, 5.10], [635, 84.00, -10.20, 5.50], [640, 85.10, -10.90, 6.70],
            [645, 83.50, -10.70, 7.00], [650, 82.75, -12.00, 8.60], [655, 82.25, -11.35, 7.95],
            [660, 82.60, -12.00, 9.20], [665, 83.75, -13.00, 9.60], [670, 84.90, -14.00, 9.80],
            [675, 83.10, -13.80, 10.00], [680, 83.30, -16.30, 10.20], [685, 76.60, -12.80, 9.25],
            [690, 73.10, -12.00, 8.30], [695, 71.90, -12.65, 8.95], [700, 74.30, -13.30, 9.60],
            [705, 75.35, -13.10, 9.05], [710, 76.40, -12.90, 8.50], [715, 69.85, -11.75, 7.75],
            [720, 65.30, -10.60, 7.00], [725, 67.35, -11.10, 7.30], [730, 71.70, -11.60, 7.60],
            [735, 74.35, -11.90, 7.80], [740, 71.00, -12.20, 8.00], [745, 71.10, -11.20, 7.35],
            [750, 65.20, -10.20, 6.70], [755, 56.45, -9.20, 5.95], [760, 47.70, -7.80, 5.20],
            [765, 58.15, -9.50, 6.30], [770, 66.80, -11.20, 7.40], [775, 66.80, -10.80, 7.10],
            [780, 65.50, -10.40, 6.80], [785, 65.00, -10.50, 6.90], [790, 66.00, -10.60, 7.00],
            [795, 63.50, -10.15, 6.70], [800, 61.00, -9.70, 6.40], [805, 57.15, -9.00, 5.95],
            [810, 53.10, -8.30, 5.50], [815, 56.30, -8.80, 5.80], [820, 58.90, -9.30, 6.10],
            [825, 60.40, -9.55, 6.30], [830, 61.90, -9.80, 6.50]
        ])
        
        # 提取波长和对应的S0, S1, S2值
        s_wavelengths = s_data[:, 0]
        s0_values = s_data[:, 1]
        s1_values = s_data[:, 2]
        s2_values = s_data[:, 3]
        
        # 插值到目标波长范围(380-780nm)
        S0 = np.interp(wavelengths, s_wavelengths, s0_values)
        S1 = np.interp(wavelengths, s_wavelengths, s1_values)
        S2 = np.interp(wavelengths, s_wavelengths, s2_values)
        
        # 公式3: S(λ) = S0(λ) + M1*S1(λ) + M2*S2(λ)
        S = S0 + M1 * S1 + M2 * S2
        
        # 归一化到560nm处
        index_560 = np.where(wavelengths == 560)[0][0]
        S_normalized = S / S[index_560]
        
        return np.column_stack((wavelengths, S_normalized))

def polygon_area(vertices):
    """多边形面积"""
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# 全局只读一次
TCS = np.loadtxt("TCS.txt")
CMFS1964 = np.loadtxt("cie1964.txt")

@numba.njit(fastmath=True, cache=True)
def get_rf_rg_fast(std_sd, sd_values, TCS, CMFS1964):
    xyz_s = np.zeros((3, 100))
    xyz_d = np.zeros((3, 100))
    for i in range(100):
        for j in range(std_sd.shape[0]):
            tcs = TCS[j, i]
            for k in range(3):
                xyz_s[k, i] += std_sd[j, 1] * tcs * CMFS1964[j + 20, k]
                xyz_d[k, i] += sd_values[j] * tcs * CMFS1964[j + 20, k]
    return xyz_s, xyz_d

def get_rf_rg(sd_data, T):
    std_sd = get_std_illuminant(T)  # shape (N, 2)
    # 构造 sd_values: 与 std_sd 波长对应的能量数组
    sd_values = np.array([sd_data[wl] if wl in sd_data.wavelengths else 0.0 for wl in std_sd[:, 0]])
    xyz_s, xyz_d = get_rf_rg_fast(std_sd, sd_values, TCS, CMFS1964)
    xyz_s_w = to_xyz1964(std_sd)
    # sd_data转为numpy数组再传入to_xyz1964
    arr = np.column_stack(
        (np.arange(380, 781), [sd_data[wl] if wl in sd_data.wavelengths else 0.0 for wl in range(380, 781)])
    )
    xyz_d_w = to_xyz1964(arr)
    xyz_s *= 100 / xyz_s_w[1] if xyz_s_w[1] != 0 else 1.0
    xyz_d *= 100 / xyz_d_w[1] if xyz_d_w[1] != 0 else 1.0
    ucs_s = np.array([colour.XYZ_to_CAM02UCS(xyz_s[:, i]) for i in range(100)]).T
    ucs_d = np.array([colour.XYZ_to_CAM02UCS(xyz_d[:, i]) for i in range(100)]).T
    d_e = np.sqrt(np.sum((ucs_s - ucs_d) ** 2, axis=0)).sum()
    Rf = 100 - 6.73 * d_e / 99
    S_s = polygon_area(ucs_s[1:3, :].T)
    S_d = polygon_area(ucs_d[1:3, :].T)
    Rg = (S_d / S_s) * 100
    return Rf, Rg

def get_melanopsin_action_spectrum():
    """生成褪黑素作用光谱（归一化）"""
    mel_action_spectrum = {}
    for wl in range(380, 781):
        if wl < 400:
            s = 0.001
        elif wl <= 420:
            s = 0.01 * np.exp(-(wl - 420) ** 2 / 800)
        elif wl <= 480:
            s = np.exp(-(wl - 480) ** 2 / 1200)
        elif wl <= 550:
            s = 0.8 * np.exp(-(wl - 480) ** 2 / 1500)
        elif wl <= 600:
            s = 0.3 * np.exp(-(wl - 480) ** 2 / 2000)
        else:
            s = 0.05 * np.exp(-(wl - 480) ** 2 / 3000)
        mel_action_spectrum[wl] = max(0.001, s)
    max_s = max(mel_action_spectrum.values())
    for wl in mel_action_spectrum:
        mel_action_spectrum[wl] /= max_s
    return mel_action_spectrum

def get_photopic_luminous_efficiency():
    """生成光视效率曲线（归一化）"""
    v_lambda = {}
    for wl in range(380, 781):
        if wl < 400:
            e = 0.0001
        elif wl <= 500:
            e = 0.01 * np.exp(-(wl - 555) ** 2 / 8000)
        elif wl <= 555:
            e = np.exp(-(wl - 555) ** 2 / 6000)
        elif wl <= 650:
            e = np.exp(-(wl - 555) ** 2 / 7000)
        else:
            e = 0.001 * np.exp(-(wl - 555) ** 2 / 10000)
        v_lambda[wl] = max(0.0001, e)
    max_e = max(v_lambda.values())
    for wl in v_lambda:
        v_lambda[wl] /= max_e
    return v_lambda

def get_d65_spd():
    """生成D65光谱分布"""
    return colour.SDS_ILLUMINANTS["D65"]

def calculate_mel_der(test_sd):
    """计算mel-DER及详细过程"""
    mel_action = get_melanopsin_action_spectrum()
    v_lambda = get_photopic_luminous_efficiency()
    d65_sd = get_d65_spd()
    test_mel_irradiance = 0.0
    test_illuminance = 0.0
    for wl in range(380, 781):
        if wl in test_sd.wavelengths:
            spd = test_sd[wl] * 1e-3
            test_mel_irradiance += spd * mel_action[wl]
            test_illuminance += spd * v_lambda[wl]
    test_illuminance *= 683
    if test_illuminance <= 0:
        return 0, {"error": "测试光源照度为零"}
    test_mel_elr = test_mel_irradiance / test_illuminance
    d65_mel_irradiance = 0.0
    d65_illuminance = 0.0
    test_total_radiance = sum(test_sd[wl] * 1e-3 for wl in test_sd.wavelengths)
    d65_scaling = test_total_radiance / 100.0
    for wl in range(380, 781):
        if wl in d65_sd.wavelengths:
            d65_value = d65_sd[wl] * d65_scaling
            d65_mel_irradiance += d65_value * mel_action[wl]
            d65_illuminance += d65_value * v_lambda[wl]
    d65_illuminance *= 683
    if d65_illuminance <= 0:
        return 0, {"error": "D65照度计算错误"}
    d65_mel_elr = d65_mel_irradiance / d65_illuminance
    if d65_mel_elr <= 0:
        return 0, {"error": "D65 mel-opic ELR为零"}
    mel_der = test_mel_elr / d65_mel_elr
    details = {
        "test_mel_irradiance": test_mel_irradiance,
        "test_illuminance": test_illuminance,
        "test_mel_elr": test_mel_elr,
        "d65_mel_irradiance": d65_mel_irradiance,
        "d65_illuminance": d65_illuminance,
        "d65_mel_elr": d65_mel_elr,
        "mel_der": mel_der
    }
    return mel_der, details

# 全局缓存
_last_W = None
_last_metrics = {}
def calc_all_metrics(W):
    global _last_W, _last_metrics
    # 如果W没变，直接返回缓存
    if _last_W is not None and np.allclose(W, _last_W):
        return _last_metrics
    sds = [sd_blue, sd_green, sd_red, sd_warm_white, sd_cool_white]
    light = weighted_sum_sds(W, sds)
    xy, XYZ, u, v = to_xyz1931(light)
    T = to_cct(xy)
    Rf, Rg = get_rf_rg(light, T)
    Duv = calculate_duv(xy)
    _last_W = np.copy(W)
    _last_metrics = {'T': T, 'Rf': Rf, 'Rg': Rg, 'Duv': Duv}
    return _last_metrics
def objective(W):
    metrics = calc_all_metrics(W)
    return -metrics['Rf']

# 定义约束为 penalty 形式

def penalty_objective(W):
    metrics = calc_all_metrics(W)
    if not np.isfinite(metrics['Rf']) or not np.isfinite(metrics['Rg']) or not np.isfinite(metrics['T']):
        return 1e6
    penalty = 0
    penalty += max(0, 5500 - metrics['T']) * 1e4      # T >= 5500
    penalty += max(0, metrics['T'] - 6500) * 1e4      # T <= 6500
    penalty += max(0, 95 - metrics['Rg']) * 1e4       # Rg >= 95
    penalty += max(0, metrics['Rg'] - 105) * 1e4      # Rg <= 105
    penalty += max(0, 88.0 - metrics['Rf']) * 1e6     # Rf >= 88
    penalty += max(0, abs(metrics['Duv']) - 0.054) * 1e4  # |DUV| <= 0.054
    penalty += abs(np.sum(W) - 1) * 1e4               # 权重和为1
    return (100 - metrics['Rf']) + penalty            # Rf越接近100越好


def constraint_sum(W):
    # 权重和为1
    return np.sum(W) - 1

def constraint_T_low(W):
    metrics = calc_all_metrics(W)
    return metrics['T'] - 5500

def constraint_T_high(W):
    metrics = calc_all_metrics(W)
    return 6500 - metrics['T']

def constraint_Rg_low(W):
    metrics = calc_all_metrics(W)
    return metrics['Rg'] - 95

def constraint_Rg_high(W):
    metrics = calc_all_metrics(W)
    return 105 - metrics['Rg']

def constraint_Rf(W):
    metrics = calc_all_metrics(W)
    return metrics['Rf'] - 88.0
def constraint_Duv(W):
    """DUV绝对值约束 <= 0.054"""
    metrics = calc_all_metrics(W)
    return 0.054 - abs(metrics['Duv'])

def weighted_sum_sds(weights, sds):
    # 直接用 numpy 合成后再构造 SpectralDistribution
    wavelengths = sds[0].wavelengths
    values = np.zeros_like(wavelengths, dtype=float)
    for w, sd in zip(weights, sds):
        values += w * np.array([sd[wl] if wl in sd.wavelengths else 0.0 for wl in wavelengths])
    return colour.SpectralDistribution(dict(zip(wavelengths, values)))


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
if __name__ == "__main__":
    bounds = [(0, 1)] * 5

    result = differential_evolution(
        penalty_objective,
        bounds=bounds,
        strategy='best1bin',
        maxiter=500,         # 可根据需要调整
        popsize=20,          # 可根据需要调整
        tol=1e-3,
        workers=-1,          # 多核加速
        updating='deferred', # 推荐配合workers
        polish=True
    )

    if result.success:
        best_W = result.x / np.sum(result.x)
        print("差分进化最优权重：", best_W)
        sds = [sd_blue, sd_green, sd_red, sd_warm_white, sd_cool_white]
        light = weighted_sum_sds(best_W, sds)
        xy, XYZ, u, v = to_xyz1931(light)
        T = to_cct(xy)
        Rf, Rg = get_rf_rg(light, T)
        mel_der, details = calculate_mel_der(light)
        print(f"最优Rf: {Rf:.2f}, T: {T:.2f}, Rg: {Rg:.2f}")
        print(f"mel-DER: {mel_der:.5f}")
    else:
        print("差分进化优化失败：", result.message)
