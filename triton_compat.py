"""triton_compat.py — Triton 3.0.0兼容层: 为缺失的Triton 3.1+ API提供monkey-patch

C500上的Triton是3.0.0+metax版本, 缺少mamba_official Mamba3内核需要的set_allocator API。
Triton 3.0.0有自己的默认内存分配器, set_allocator只是覆盖它, 所以用空操作即可。
"""
import triton

if not hasattr(triton, 'set_allocator'):
    _alloc_fn_ref = None

    def _set_allocator(fn):
        """空操作: Triton 3.0.0使用默认分配器, 不需要显式设置。"""
        global _alloc_fn_ref
        _alloc_fn_ref = fn

    triton.set_allocator = _set_allocator
    # 标记为已patch
    triton._set_allocator_patched = True
