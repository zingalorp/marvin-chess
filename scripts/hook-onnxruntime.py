# PyInstaller hook for onnxruntime
# This ensures all required DLLs are properly collected

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

# Collect everything from onnxruntime
datas, binaries, hiddenimports = collect_all('onnxruntime')

# Also collect dynamic libraries
binaries += collect_dynamic_libs('onnxruntime')

# Add any missing hidden imports
hiddenimports += [
    'onnxruntime.capi',
    'onnxruntime.capi.onnxruntime_pybind11_state',
    'onnxruntime.capi._pybind_state',
]
