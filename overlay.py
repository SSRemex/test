"""Windows-only experimental click-through overlay."""

import ctypes
import sys
import tkinter as tk
from ctypes import wintypes
from tkinter import messagebox

from PIL import ImageTk

from watermark import WATERMARK, make_overlay, text_mask


def enable_dpi_awareness():
    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except AttributeError:
        user32.SetProcessDPIAware()


def monitors():
    rectangles = []
    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM
    )

    @callback_type
    def collect(monitor, device_context, rectangle, data):
        bounds = rectangle.contents
        rectangles.append((bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top))
        return True

    if not ctypes.windll.user32.EnumDisplayMonitors(None, None, collect, 0):
        raise ctypes.WinError()
    return rectangles


def run_overlay(identifier, alpha, stop_after_ms=None):
    if sys.platform != "win32":
        raise RuntimeError("屏幕覆盖层目前仅支持 Windows")
    if not 0.004 <= alpha <= 0.04:
        raise ValueError("透明度必须在 0.004 到 0.04 之间")
    text_mask(identifier)
    enable_dpi_awareness()
    user32 = ctypes.windll.user32
    get_parent = user32.GetParent
    get_parent.argtypes = [wintypes.HWND]
    get_parent.restype = wintypes.HWND
    get_style = user32.GetWindowLongPtrW
    get_style.argtypes = [wintypes.HWND, ctypes.c_int]
    get_style.restype = ctypes.c_ssize_t
    set_style = user32.SetWindowLongPtrW
    set_style.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    set_style.restype = ctypes.c_ssize_t
    user32.SetWindowPos.argtypes = [
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT
    ]
    user32.SetWindowPos.restype = wintypes.BOOL

    root = tk.Tk()
    root.title("暗水印验证控制台")
    root.geometry("420x165")
    root.resizable(False, False)
    tk.Label(root, text=f"测试编号：{identifier}\n关闭此窗口即可停止水印。", pady=12).pack()
    tk.Label(root, text="实验版：显示器变化后请停止并重新启动。").pack()
    tk.Button(root, text="停止水印", command=root.destroy, width=18).pack(pady=10)
    root.protocol("WM_DELETE_WINDOW", root.destroy)

    windows = []
    images = []
    for left, top, width, height in monitors():
        window = tk.Toplevel(root)
        window.withdraw()
        window.overrideredirect(True)
        window.attributes("-topmost", True)
        window.attributes("-transparentcolor", "#00ff00")
        window.attributes("-alpha", alpha)
        window.geometry(f"{width}x{height}+0+0")
        image = ImageTk.PhotoImage(make_overlay(identifier, width, height))
        tk.Label(window, image=image, borderwidth=0, highlightthickness=0).pack()
        window.update_idletasks()
        handle = get_parent(window.winfo_id()) or window.winfo_id()
        extended_style = get_style(handle, -20)
        required_style = 0x20 | 0x80 | 0x08000000
        set_style(handle, -20, extended_style | required_style)
        if get_style(handle, -20) & required_style != required_style:
            root.destroy()
            raise RuntimeError("无法设置鼠标穿透和不激活窗口，已停止启动")
        window.deiconify()
        window.update_idletasks()
        if not user32.SetWindowPos(handle, wintypes.HWND(-1), left, top, width, height, 0x0010 | 0x0040):
            root.destroy()
            raise ctypes.WinError()
        windows.append(window)
        images.append(image)
    if not windows:
        root.destroy()
        raise RuntimeError("未找到显示器")
    root.lift()
    if stop_after_ms is not None:
        root.after(stop_after_ms, root.destroy)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        root.destroy()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="临时运行 Windows 暗水印覆盖层")
    parser.add_argument("--id", default=WATERMARK, help=f"16 位数字测试编号，默认 {WATERMARK}")
    parser.add_argument("--alpha", type=float, default=0.004, help="覆盖层透明度，默认 0.004")
    arguments = parser.parse_args()
    try:
        run_overlay(arguments.id, arguments.alpha)
    except Exception as error:
        messagebox.showerror("暗水印启动失败", str(error))
        raise


if __name__ == "__main__":
    main()
