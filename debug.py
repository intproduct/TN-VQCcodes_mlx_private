import mlx.core as mx
import numpy as np
import math
from typing import List, Tuple, Optional, Union, Iterable
import matplotlib.pyplot as plt
import os

def compare_dicts(
        dict1: dict,
        dict2: dict,
        name1: str = 'dict1',
        name2: str = 'dict2',
        *,
        strict: bool = True,
        report_diff: bool = True,
        compare_values: bool = False
) -> bool:
    #比较两个字典是否一致
    #开启strict模式时，如果dict1和dict2中存在不同的key，则返回False
    #开启report_diff模式时，如果dict1和dict2中存在不同的key，则打印出不同的key
    #开启compare_values模式时，如果dict1和dict2中存在不同的key，则打印出不同的key和对应的值
    #如果dict1和dict2中存在不同的key，则返回False
    same = True

    for key in dict1:
        if key not in dict2:
            if report_diff:
                print(f"{key} in {name1} but not in {name2}")
            elif compare_values and dict1[key] != dict2[key]:
                if report_diff:
                    print(f"{key}: {name1} = {dict1[key]!r} != {name2} = {dict2[key]!r}")
            same = False

    if strict:
        for key in dict2:
            if key not in dict1:
                if report_diff:
                    print(f"{key} in {name2} but not in {name1}")
                same = False

    return same

def has_common_elements(iter1: Iterable, iter2: Iterable) -> bool:
    if not (isinstance(iter1, Iterable) and isinstance(iter2, Iterable)):
        raise TypeError("iter1 and iter2 must be iterable")
    
    try:
        return bool(set(iter1) & set(iter2))
    except TypeError:
        for item in iter1:
            if item in iter2:
                return True
        return False


def fprint(content: str, file: Optional[str] = None, 
           print_screen: bool = True, append: bool = True) -> None:
    if file is not None:
        mode = 'a' if append else 'w'
        with open(file, mode, encoding='utf-8') as f:
            f.write(content + '\n')
            
    if print_screen:
        print(content)

def print_dict(
        dict1: dict, 
        key: Optional[Union[str, list, tuple]] = None, 
        welcome: str ='',
        style_sep: str = '',
        end: str = '\n',
        file: Optional[str] = None,
        print_screen: bool = True,
        append: bool = True
        ) -> str:
    express = welcome

    if key is None:
        parts = [f"{k}: {v}" for k, v in dict1.items()]
        express += end.join(parts)+ (end if parts else '')
    else:
        if isinstance(key, str):
            express += f"{key.capitalize()}{style_sep}{dict1[key]}"
        else:
            key = list(key)
            parts = [f"{k.capitalize()}{style_sep}{dict1[k]}" for k in key]
            express += end.join(parts)
    fprint(express, file, print_screen, append)
    return express

'''def print_process_bar():
    return ''' #可以直接用tqdm替代

def debug_viz(
    data: Union[np.array, mx.array, List],
    *extra_y,  # 仅曲线模式使用
    mode = None,          # 'line', 'image', or auto
    marker = '.', 
    linestyle = '-', 
    xlabel = None, 
    ylabel = None, 
    legend = None,
    titles = None,
    grid_shape = None,    # (rows, cols) for images
    cmap = 'coolwarm',
    img_size = None,
    save_path = None,
    show = True
):
    """
    ⚠️ 调试专用：智能可视化函数（自动区分曲线图 vs 图像网格）
    
    支持：
      - 1D/2D 张量 → 折线图（多曲线）
      - 2D/3D/4D 张量 → 图像网格
      - 自动处理 torch.Tensor / mx.array / numpy
    
    Args:
        data: 主要数据（张量或数组）
        *extra_y: 额外的 y 序列（仅当画曲线时）
        mode: 强制模式 'line' 或 'image'，否则自动推断
        ... 其他参数根据模式生效
    """
    
    # --- 通用预处理：转为 numpy ---
    def to_numpy(x):
        if hasattr(x, 'tolist'): # mx.array 等
            x = np.array(x.tolist())
        else:
            x = np.asarray(x)
        return x
    
    data = to_numpy(data)
    extra_y = [to_numpy(y) for y in extra_y]

    # --- 模式推断 ---
    if mode is None:
        if extra_y or (data.ndim == 1) or (data.ndim == 2 and data.shape[0] > 10):
            # 启发式：如果传了 extra_y，或 data 是长向量 → 当作曲线
            mode = 'line'
        else:
            mode = 'image'

    # --- 分支执行 ---
    if mode == 'line':
        _plot_lines(
            data, extra_y, marker, linestyle, xlabel, ylabel, legend, show, save_path
        )
    elif mode == 'image':
        _plot_images(
            data, titles, grid_shape, cmap, img_size, show, save_path
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


# --- 内部辅助函数（不暴露给用户）---

def _plot_lines(
        x, 
        y_list, 
        marker, 
        linestyle, 
        xlabel, 
        ylabel, 
        legend, 
        show, 
        save_path):
    fig, ax = plt.subplots()
    
    if not y_list:  # 只传了 x，当作 y
        y_data = [x]
        x_data = np.arange(len(x))
    else:
        y_data = [x] + y_list  # x 是公共 x 轴
        x_data = x
    
    # 扩展样式
    markers = [marker] * len(y_data) if isinstance(marker, str) else marker
    linestyles = [linestyle] * len(y_data) if isinstance(linestyle, str) else linestyle

    lines = []
    for y, m, ls in zip(y_data, markers, linestyles):
        if y is x_data:
            continue  # 跳过 x 本身（当 y_list 非空时）
        line, = ax.plot(x_data, y, marker=m, linestyle=ls)
        lines.append(line)
    
    if not y_list:  # 单曲线情况
        lines = [ax.plot(x_data, x, marker=marker, linestyle=linestyle)[0]]
    
    if legend:
        ax.legend(lines, legend)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def _plot_images(imgs, titles, grid_shape, cmap, img_size, show, save_path):
    if imgs.ndim == 2:
        # 单张图像
        plt.figure()
        plt.imshow(imgs, cmap=cmap)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        return

    # 批量图像
    if imgs.ndim == 3:
        num_imgs = imgs.shape[0]
    else:
        raise ValueError("Image batch must be 3D (HWC or CHW not supported here)")

    # 推断网格
    if grid_shape is None:
        cols = math.ceil(math.sqrt(num_imgs))
        rows = math.ceil(num_imgs / cols)
    else:
        rows, cols = grid_shape
        if rows == -1:
            rows = math.ceil(num_imgs / cols)
        elif cols == -1:
            cols = math.ceil(num_imgs / rows)

    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(num_imgs):
        plt.subplot(rows, cols, i + 1)
        img = imgs[i]
        if img_size is not None:
            img = img.reshape(img_size)
        if img.ndim == 2:
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(img)
        if titles is not None and i < len(titles):
            plt.title(str(titles[i]))
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()

def search_file(path, exp):
    import re
    content = os.listdir(path)
    exp = re.compile(exp)
    result = list()
    for x in content:
        if re.match(exp, x):
            result.append(os.path.join(path, x))
    return result

def save_mlx_checkpoint(path, file, data, names, append=False):
    path_file = os.path.join(path, file) if path else file
    os.makedirs(os.path.dirname(path_file), exist_ok=True)
    
    # 转为 numpy
    arrays = {name: arr if np.isscalar(arr) else np.array(arr) 
              for name, arr in zip(names, data)}
    
    if append and os.path.exists(path_file):
        old = np.load(path_file)
        arrays = {**dict(old), **arrays}  # new overrides old
        old.close()
    
    np.savez_compressed(path_file, **arrays)

def load_mlx(path_file, names=None, return_tuple=True):
    """
    从 .npz 文件加载 MLX 数组（兼容调试用 save_mlx）。
    
    接口设计与 torch 版 load 一致（无 device 参数）。
    
    Args:
        path_file (str): .npz 文件路径
        names (str, list, tuple, or None): 
            - None: 加载全部键
            - str: 加载单个键（直接返回 mx.array）
            - list/tuple: 按 names 顺序返回对应 mx.array 的元组（缺失键返回 None）
        return_tuple (bool): 当 names=None 且多键时，是否返回 tuple（否则返回 dict）
    
    Returns:
        - 单键: mx.array（或 None）
        - 多键: tuple（顺序与 names 一致）或 dict
        - 文件不存在: None
    """
    if not os.path.isfile(path_file):
        return None

    try:
        with np.load(path_file, allow_pickle=False) as npz_file:
            # 转为 dict，避免 npz_file 的上下文限制
            tmp = {k: v for k, v in npz_file.items()}
    except Exception as e:
        raise ValueError(f"Failed to load {path_file}: {e}")

    # 将 numpy 数组转为 mlx array
    tmp_mx = {k: mx.array(v) for k, v in tmp.items()}

    if names is None:
        if len(tmp_mx) == 1:
            return next(iter(tmp_mx.values()))
        else:
            return tuple(tmp_mx.values()) if return_tuple else tmp_mx

    elif isinstance(names, str):
        return tmp_mx.get(names, None)

    elif isinstance(names, (list, tuple)):
        return tuple(tmp_mx.get(name, None) for name in names)

    else:
        return None

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)