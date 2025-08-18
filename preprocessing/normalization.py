import numpy as np
from typing import Optional, Tuple
def robust_winsor_scale(
    arr: np.ndarray,
    win_quant: Tuple[float, float] = (0.01, 0.99),
    scale_quant: Tuple[float, float] = (0.25, 0.75),
    axis: Optional[int] = None,
    epsilon: float = 1e-8,
    inplace: bool = False
) -> np.ndarray:
    """
    Single function to implement data shrinking + robust standardization
    
    Parameters:
        arr : 输入 numpy 数组（支持任意维度） (Input numpy array of any dimension)
        win_quant : 缩尾分位区间 (lower, upper) (window dimensions)
        scale_quant : 标准化分位区间 (lower, upper)
        axis : 计算轴向 (None 为全局处理)
        epsilon : 防止除零的极小值 
        inplace : 是否原地修改
    
    Returns:
        处理后的 numpy 数组
    
    Features:
        - 自动维度广播
        - 数值稳定性保障
        - 内存效率优化
        - 异常值双重防护
    """
    # Validate parameters
    if not (0 <= win_quant[0] < win_quant[1] <= 1):
        raise ValueError("Winsorization needs 0 ≤ lower < upper ≤ 1")
    if not (0 <= scale_quant[0] < scale_quant[1] <= 1):
        raise ValueError("Normalization needs 0 ≤ lower < upper ≤ 1")

    if np.isnan(arr).any():
        raise ValueError("Input contains NaNs. Please handle them before normalization.")

    # inplace or not
    arr = arr if inplace else arr.copy()

    # perform winsorization
    lower_win, upper_win = np.quantile(arr, win_quant, axis=axis, keepdims=True)
    win_range = upper_win - lower_win
    lower_clip = lower_win - 0.1*win_range
    upper_clip = upper_win + 0.1*win_range
    np.clip(arr, lower_clip, upper_clip, out=arr)

    # Robust standardization
    q1, q3 = np.quantile(arr, scale_quant, axis=axis, keepdims=True)
    iqr = q3 - q1
    median = np.median(arr, axis=axis, keepdims=True)

    # Ensure IQR is not too small to avoid division issues
    iqr = np.where(iqr < epsilon, epsilon, iqr)

    # Standardize to approximate standard normal distribution (unit std)
    arr -= median
    arr /= iqr / 1.349  # IQR to std conversion factor for normal distribution

    return arr