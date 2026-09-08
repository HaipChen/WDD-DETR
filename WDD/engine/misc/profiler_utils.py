"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from typing import Tuple
import torch
import torch.nn as nn

# 尝试导入calflops，如果不可用则使用备选方法
try:
    from calflops import calculate_flops
    CALFLOPS_AVAILABLE = True
except ImportError:
    CALFLOPS_AVAILABLE = False


def stats(
    cfg,
    input_shape: Tuple = (1, 3, 640, 640),
) -> Tuple[int, dict]:
    """
    计算模型的计算量和参数量

    Args:
        cfg: 配置对象
        input_shape: 输入张量形状 (batch, channels, height, width)

    Returns:
        tuple: (参数量, 统计信息字典)
    """
    base_size = cfg.train_dataloader.collate_fn.base_size
    input_shape = (1, 3, base_size, base_size)

    # 获取模型
    model_for_info = copy.deepcopy(cfg.model).deploy()

    # 获取FLOPs计算方法，默认为'auto'
    flops_method = getattr(cfg, 'flops_cal_method', 'calflops')

    # 根据选择的方法计算FLOPs
    if flops_method == 'calflops' or (flops_method == 'auto' and CALFLOPS_AVAILABLE):
        # 使用calflops方法
        flops, macs, params = calculate_flops_with_fallback(model_for_info, input_shape)
    elif flops_method == 'fvcore': #flops、MACs x2=calflops
        # 使用fvcore方法
        flops, macs = calculate_flops_fvcore(model_for_info, input_shape)
    elif flops_method == 'thop':  #flops、MACs x2=calflops
        # 使用thop方法
        flops, macs = calculate_flops_thop(model_for_info, input_shape)
    elif flops_method == 'torchprof':
        # 使用torchprofiler方法
        flops, macs = calculate_flops_torchprofiler(model_for_info, input_shape)
    elif flops_method == 'manual':
        # 使用手动计算方法
        flops, macs = calculate_flops_manual(model_for_info, input_shape)
    elif flops_method == 'auto' and not CALFLOPS_AVAILABLE:
        # 自动选择方法（calflops不可用时）
        flops, macs = calculate_flops_auto(model_for_info, input_shape)
    else:
        raise ValueError(f"不支持的FLOPs计算方法: {flops_method}")

    # 格式化输出
    flops_str = format_number(flops)
    macs_str = format_number(macs)

    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info

    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" % (flops_str, macs_str, params)}


def calculate_flops_with_fallback(model, input_shape):
    """使用calflops，失败时自动回退到其他方法"""
    if not CALFLOPS_AVAILABLE:
        print("警告: calflops不可用，尝试使用备选方法...")
        flops, macs = calculate_flops_auto(model, input_shape)
        return flops, macs, 0  # 返回三个值

    try:
        flops, macs, params = calculate_flops(
            model=model,
            input_shape=input_shape,
            output_as_string=False,
            output_precision=4,
            print_detailed=False
        )
        return flops, macs, params
    except Exception as e:
        print(f"警告: calflops计算失败: {e}")
        print("尝试使用备选方法...")
        try:
            flops, macs, params = calculate_flops_auto(model, input_shape)
            return flops, macs,params  # 确保返回三个值
        except Exception as e2:
            print(f"备选方法也失败: {e2}")
            # 返回默认值
            return 0, 0, 0


def calculate_flops_auto(model, input_shape):
    """自动选择可用的最佳计算方法"""
    # 按优先级尝试各种方法
    methods_to_try = [
        ('fvcore', calculate_flops_fvcore),
        ('thop', calculate_flops_thop),
        ('torchprof', calculate_flops_torchprofiler),
        ('manual', calculate_flops_manual)
    ]

    for method_name, method_func in methods_to_try:
        try:
            flops, macs = method_func(model, input_shape)
            print(f"使用 {method_name} 方法计算成功")
            return flops, macs
        except Exception as e:
            print(f"{method_name} 方法失败: {e}")
            continue

    # 所有方法都失败，使用最简单的估算
    print("所有方法均失败，使用基础估算")
    return calculate_flops_basic(model, input_shape)


def calculate_flops_fvcore(model, input_shape):
    """使用fvcore库计算FLOPs"""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        raise ImportError("请安装fvcore库: pip install fvcore")

    model.eval()
    device = next(model.parameters()).device
    input_tensor = torch.randn(input_shape, device=device)

    flop_counter = FlopCountAnalysis(model, (input_tensor,))
    flops = flop_counter.total()
    macs = flops / 2  # MACs通常是FLOPs的一半

    return flops, macs


def calculate_flops_thop(model, input_shape):
    """使用thop库计算FLOPs"""
    try:
        from thop import profile
    except ImportError:
        raise ImportError("请安装thop库: pip install thop")

    model.eval()
    device = next(model.parameters()).device
    input_tensor = torch.randn(input_shape, device=device)

    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    macs = flops / 2

    return flops, macs


def calculate_flops_torchprofiler(model, input_shape):
    """使用PyTorch Profiler计算FLOPs"""
    model.eval()
    device = next(model.parameters()).device
    input_tensor = torch.randn(input_shape, device=device)

    try:
        with torch.no_grad():
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=True,
                with_flops=True
            ) as prof:
                _ = model(input_tensor)

        flops = 0
        for event in prof.key_averages():
            if hasattr(event, "flops") and event.flops > 0:
                flops += event.flops
        macs = flops / 2

        if flops == 0:
            # 如果没有获取到FLOPs，使用估算
            raise ValueError("未能从profiler获取FLOPs信息")

        return flops, macs
    except Exception as e:
        raise RuntimeError(f"PyTorch Profiler计算失败: {e}")


def calculate_flops_manual(model, input_shape):
    """
    更准确的手动FLOPs计算方法
    不依赖profiler，直接分析模型结构
    """
    model.eval()
    device = next(model.parameters()).device

    # 注册hook来捕获每一层的计算
    flops_total = 0

    def conv2d_flops_hook(module, input, output):
        nonlocal flops_total
        # 计算Conv2d的FLOPs
        # FLOPs = batch_size * output_h * output_w * out_channels * kernel_h * kernel_w * in_channels
        batch_size, in_channels, in_h, in_w = input[0].shape
        out_channels, _, kernel_h, kernel_w = module.weight.shape
        output_h, output_w = output.shape[2], output.shape[3]

        # 每个输出位置的计算量
        flops_per_position = kernel_h * kernel_w * in_channels
        # 总计算量
        total_flops = batch_size * output_h * output_w * out_channels * flops_per_position

        # 如果包含偏置，加上偏置计算
        if module.bias is not None:
            total_flops += batch_size * output_h * output_w * out_channels

        flops_total += total_flops

    def linear_flops_hook(module, input, output):
        nonlocal flops_total
        # 计算Linear层的FLOPs
        # FLOPs = batch_size * in_features * out_features
        batch_size = input[0].shape[0]
        in_features = module.in_features
        out_features = module.out_features

        total_flops = batch_size * in_features * out_features

        # 如果包含偏置，加上偏置计算
        if module.bias is not None:
            total_flops += batch_size * out_features

        flops_total += total_flops

    def batch_norm_flops_hook(module, input, output):
        nonlocal flops_total
        # BatchNorm的计算量较小，通常可以忽略或简单估算
        # 每个通道2个操作：减均值，除以标准差
        batch_size, channels, h, w = input[0].shape
        flops_total += batch_size * channels * h * w * 2

    hooks = []

    # 为不同类型的层注册hook
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hook = module.register_forward_hook(conv2d_flops_hook)
            hooks.append(hook)
        elif isinstance(module, torch.nn.Linear):
            hook = module.register_forward_hook(linear_flops_hook)
            hooks.append(hook)
        elif isinstance(module, torch.nn.BatchNorm2d):
            hook = module.register_forward_hook(batch_norm_flops_hook)
            hooks.append(hook)

    # 运行前向传播
    input_tensor = torch.randn(input_shape, device=device)
    with torch.no_grad():
        _ = model(input_tensor)

    # 移除所有hook
    for hook in hooks:
        hook.remove()

    # 如果计算量仍然为0，使用经验公式
    if flops_total == 0:
        print("Hook方法未能捕获计算量，使用经验公式")
        total_params = sum(p.numel() for p in model.parameters())
        input_h, input_w = input_shape[2], input_shape[3]
        flops_total = total_params * input_h * input_w * 0.2

    macs_total = flops_total / 2
    return flops_total, macs_total


def calculate_flops_basic(model, input_shape):
    """基础估算方法"""
    total_params = sum(p.numel() for p in model.parameters())
    input_h, input_w = input_shape[2], input_shape[3]
    flops_estimate = total_params * input_h * input_w * 0.2
    macs_estimate = flops_estimate / 2
    return flops_estimate, macs_estimate


def format_number(num):
    """
    格式化数字为易读的字符串
    例如: 1500000 -> "1.5M"
    """
    if num >= 1e12:  # 万亿
        return f"{num/1e12:.4f}T"
    elif num >= 1e9:  # 十亿
        return f"{num/1e9:.4f}G"
    elif num >= 1e6:  # 百万
        return f"{num/1e6:.4f}M"
    elif num >= 1e3:  # 千
        return f"{num/1e3:.4f}K"
    else:
        return f"{num:.0f}"