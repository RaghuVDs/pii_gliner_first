"""One-shot check: can onnxruntime load the GLiNER ONNX model on CUDA?

Run with the cu12 nvidia lib dirs on LD_LIBRARY_PATH. Prints OK/FAIL.
"""
import glob
import os

import onnxruntime as ort

print("available providers:", ort.get_available_providers())

pattern = os.path.expanduser(
    "~/.cache/huggingface/hub/models--knowledgator--gliner-pii-large-v1.0"
    "/snapshots/*/onnx/model_quint8.onnx"
)
matches = glob.glob(pattern)
if not matches:
    print("FAIL — no cached ONNX file found at", pattern)
    raise SystemExit(1)

onnx_path = matches[0]
print("model:", onnx_path)

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

try:
    s = ort.InferenceSession(
        onnx_path, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
except Exception as e:
    print("FAIL — session creation raised:", repr(e))
    raise SystemExit(1)

active = s.get_providers()
print("active providers:", active)
print("OK — CUDA EP is live" if active[0] == "CUDAExecutionProvider" else "FAIL — fell back to CPU")
