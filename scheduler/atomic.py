"""跨平台原子 CAS。"""

import ctypes
import ctypes.util
import sys
import platform

_cas32 = None
_cas64 = None
_windows_code_pages = []


def _make_windows_cas(code, pointer_type, value_type):
    """创建使用 Windows 调用约定的 x86/x64 compare-and-exchange 封装。"""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    virtual_alloc = kernel32.VirtualAlloc
    virtual_alloc.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    virtual_alloc.restype = ctypes.c_void_p

    virtual_protect = kernel32.VirtualProtect
    virtual_protect.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    virtual_protect.restype = ctypes.c_bool

    flush_instruction_cache = kernel32.FlushInstructionCache
    flush_instruction_cache.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    flush_instruction_cache.restype = ctypes.c_bool

    # Allocate writable memory first, then switch it to execute/read (W^X).
    address = virtual_alloc(None, len(code), 0x3000, 0x04)
    if not address:
        raise ctypes.WinError(ctypes.get_last_error())
    ctypes.memmove(address, code, len(code))
    old_protection = ctypes.c_ulong()
    if not virtual_protect(address, len(code), 0x20, ctypes.byref(old_protection)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not flush_instruction_cache(
        kernel32.GetCurrentProcess(), address, len(code)
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    _windows_code_pages.append(address)
    return ctypes.CFUNCTYPE(
        value_type, ctypes.POINTER(pointer_type), value_type, value_type
    )(address)


def _init_windows_atomic():
    global _cas32, _cas64

    if ctypes.sizeof(ctypes.c_void_p) == 8:
        # Windows x64: RCX=destination, RDX=exchange, R8=comparand.
        _cas32 = _make_windows_cas(
            b"\x44\x89\xc0\xf0\x0f\xb1\x11\xc3",
            ctypes.c_int32,
            ctypes.c_int32,
        )
        _cas64 = _make_windows_cas(
            b"\x49\x8b\xc0\xf0\x48\x0f\xb1\x11\xc3",
            ctypes.c_int64,
            ctypes.c_int64,
        )
        return

    # Windows x86 cdecl: return address, destination, exchange, comparand.
    _cas32 = _make_windows_cas(
        b"\x8b\x44\x24\x0c\x8b\x4c\x24\x04\x8b\x54\x24\x08\xf0\x0f\xb1\x11\xc3",
        ctypes.c_int32,
        ctypes.c_int32,
    )
    # cmpxchg8b requires EBX/ECX for the exchange value, so preserve EBX.
    _cas64 = _make_windows_cas(
        b"\x53\x57\x8b\x7c\x24\x0c\x8b\x5c\x24\x10\x8b\x4c\x24\x14"
        b"\x8b\x44\x24\x18\x8b\x54\x24\x1c\xf0\x0f\xc7\x0f\x5f\x5b\xc3",
        ctypes.c_int64,
        ctypes.c_int64,
    )

def _init_atomic():
    global _cas32, _cas64
    system = platform.system()

    if system == "Windows":
        _init_windows_atomic()
        return

    lib_names = ['libatomic.so.1', 'libatomic.so', 'libatomic.dylib']
    lib = None
    for name in lib_names:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue

    if lib is None:
        raise RuntimeError("libatomic not found")

    try:
        _cas32 = lib.__atomic_compare_exchange_4
        _cas32.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_bool,
            ctypes.c_int,
            ctypes.c_int,
        ]
        _cas32.restype = ctypes.c_bool
    except AttributeError:
        raise RuntimeError("__atomic_compare_exchange_4 not found in libatomic")

    try:
        _cas64 = lib.__atomic_compare_exchange_8
        _cas64.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_bool,
            ctypes.c_int,
            ctypes.c_int,
        ]
        _cas64.restype = ctypes.c_bool
    except AttributeError:
        pass

_init_atomic()

def cas32(addr_ptr, expected, new):
    """32-bit 原子 CAS"""
    if platform.system() == "Windows":
        old = _cas32(addr_ptr, new, expected)
        return old == expected
    expected_ptr = ctypes.pointer(ctypes.c_int32(expected))
    return _cas32(addr_ptr, expected_ptr, new, False, 0, 0)

def cas64(addr_ptr, expected, new):
    """64-bit 原子 CAS"""
    if platform.system() == "Windows":
        old = _cas64(addr_ptr, new, expected)
        return old == expected
    expected_ptr = ctypes.pointer(ctypes.c_int64(expected))
    return _cas64(addr_ptr, expected_ptr, new, False, 0, 0)
