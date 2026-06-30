"""检查绘图环境：matplotlib + 中文字体。"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

print("=== 绘图环境检查 ===")
try:
    import matplotlib
    print(f"matplotlib: {matplotlib.__version__} ✅")
except ImportError as e:
    print(f"matplotlib: 缺失 ❌ ({e})")
    sys.exit(1)

try:
    import numpy as np
    print(f"numpy: {np.__version__} ✅")
except ImportError as e:
    print(f"numpy: 缺失 ❌ ({e})")
    sys.exit(1)

# 检查中文字体
import matplotlib.font_manager as fm
fonts = {f.name for f in fm.fontManager.ttflist}
cjk_candidates = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
                  "Source Han Sans SC", "SimHei", "Microsoft YaHei", "DejaVu Sans"]
print("\n=== 中文字体检查 ===")
found_cjk = False
for name in cjk_candidates:
    if name in fonts:
        print(f"  {name}: 可用 ✅")
        found_cjk = True
    else:
        print(f"  {name:25s}: 缺失")

if not found_cjk:
    print("\n⚠️ 未找到中文字体，中文标签将显示为方框。")
    print("  尝试安装: sudo apt install fonts-noto-cjk 或 fonts-wqy-zenhei")
    print("  或图表改用英文标签")
else:
    print(f"\n✅ 找到中文字体，可使用中文标签")

# 列出所有可用字体（前 20 个）
print(f"\n=== 可用字体总数: {len(fonts)} ===")
