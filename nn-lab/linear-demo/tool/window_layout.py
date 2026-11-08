"""将 macOS 原生 Matplotlib 窗口按屏幕可用宽度横向排列。"""

import ctypes
import sys


class Point(ctypes.Structure):
    _fields_ = [('x', ctypes.c_double), ('y', ctypes.c_double)]


class Size(ctypes.Structure):
    _fields_ = [('width', ctypes.c_double), ('height', ctypes.c_double)]


class Rect(ctypes.Structure):
    _fields_ = [('origin', Point), ('size', Size)]


def arrange_macos_windows(titles: list[str]) -> None:
    """使用系统自带的 Objective-C 接口，仅移动标题匹配的绘图窗口。"""
    if sys.platform != 'darwin' or not titles:
        return

    # MacOSX 后端未暴露窗口定位接口，直接调用当前进程内的 AppKit 窗口。
    objc: ctypes.CDLL = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p

    def send(receiver: int, selector: str, result: type = ctypes.c_void_p,
             argtypes: tuple = (), args: tuple = ()):
        # 每种消息使用独立的函数签名，避免混用指针、整数和矩形的返回类型。
        address: int = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        # Intel 的大结构体返回需使用 stret；Apple Silicon 直接使用 objc_msgSend。
        if result is Rect and sys.platform == 'darwin':
            import platform
            if platform.machine() == 'x86_64':
                function = ctypes.CFUNCTYPE(None, ctypes.POINTER(Rect), ctypes.c_void_p,
                                            ctypes.c_void_p)(('objc_msgSend_stret', objc))
                rectangle: Rect = Rect()
                function(ctypes.byref(rectangle), receiver, objc.sel_registerName(selector.encode()))
                return rectangle
        function = ctypes.CFUNCTYPE(result, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(address)
        return function(receiver, objc.sel_registerName(selector.encode()), *args)

    application: int = send(objc.objc_getClass(b'NSApplication'), 'sharedApplication')
    windows: int = send(application, 'windows')
    matching: dict[str, int] = {}
    for index in range(send(windows, 'count', ctypes.c_ulong)):
        window: int = send(windows, 'objectAtIndex:', argtypes=(ctypes.c_ulong,), args=(index,))
        title_object: int = send(window, 'title')
        title_bytes: bytes = send(title_object, 'UTF8String', ctypes.c_char_p) or b''
        title: str = title_bytes.decode('utf-8')
        if title in titles:
            matching[title] = window
    if len(matching) != len(titles):
        return

    screen: int = send(matching[titles[0]], 'screen')
    visible: Rect = send(screen, 'visibleFrame', Rect)  # 排除菜单栏和 Dock。
    gap: float = 12.0
    width: float = (visible.size.width - gap * (len(titles) + 1)) / len(titles)
    height: float = min(visible.size.height - 2 * gap, width * 0.8)
    for index, title in enumerate(titles):
        frame: Rect = Rect(
            Point(visible.origin.x + gap + index * (width + gap),
                  visible.origin.y + (visible.size.height - height) / 2),
            Size(width, height),
        )
        send(matching[title], 'setFrame:display:', None,
             (Rect, ctypes.c_bool), (frame, True))
