import json

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from torchvision import transforms


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


if __name__ == "__main__":
    img = (
        transforms.PILToTensor()(Image.open("./image224.jpg").convert("RGB"))
        .unsqueeze(0)
        .float()
        / 255.0
    )

    model_fp32_path = "clip_rn50.onnx"
    model_onnx = onnx.load(model_fp32_path)
    onnx.checker.check_model(model_onnx)

    ort_sess = ort.InferenceSession(model_fp32_path)

    # Compute the onnx embedding
    ort_inputs = {ort_sess.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_sess.run(None, ort_inputs)[0]
    # print(ort_outs)

    # Load the json exported embedding
    with open("ebd.json") as f:
        d = json.load(f)

    # Compute the cosine similarity between them
    js_ebd = np.array(
        [d[str(i)] for i in range(len(d))]
    )  # convert the json dict to a list
    py_ebd = ort_outs[0]

    print(
        "cos sim:",
        (py_ebd / np.linalg.norm(py_ebd)) @ (js_ebd / np.linalg.norm(js_ebd)),
    )
