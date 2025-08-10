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
    """黑体归一化光谱"""
    Lambda_nm = np.asarray(Lambda_nm, dtype=float)
    exponent = 1.4388e7 / (Lambda_nm * T)
    return Lambda_nm ** (-5) / (np.exp(exponent) - 1)

def get_std_illuminant(T):
    """标准光源归一化光谱"""
    wavelengths = np.arange(380, 781, dtype=float)
    values = L(wavelengths, T) / L(560, T)
    return np.column_stack((wavelengths, values))

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
    _last_W = np.copy(W)
    _last_metrics = {'T': T, 'Rf': Rf, 'Rg': Rg}
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
    return metrics['Rf'] - 87.0

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
