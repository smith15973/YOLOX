import torch
from yolox.exp import get_exp

# Load experiment
exp = get_exp("exps/example/custom/yolox_nano_basketball.py", None)
model = exp.get_model()
model.head.decode_in_inference = True  # <-- Add this line!

# Load checkpoint
ckpt = torch.load("models/best_nano_ckpt.pth", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 640, 640)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "best_nano_model_decoded.onnx",
    opset_version=11,
    input_names=['images'],
    output_names=['output']
)

print("Export successful! Saved to best_nano_model_decoded.onnx")