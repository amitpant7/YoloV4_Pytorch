import torch


def to_onnx(model, input_tensor, onnx_file_path, opset_version=11):
    """
    Convert a PyTorch model to ONNX format.

    Parameters:
    - model: The PyTorch model to be converted.
    - input_tensor: A sample input tensor to the model. It should have the same shape as the model's expected input.
    - onnx_file_path: The path where the ONNX model will be saved.
    - opset_version: The ONNX opset version to use. Default is 11.

    Returns:
    - None
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Export the model
    torch.onnx.export(
        model,  # The PyTorch model
        input_tensor,  # A sample input tensor
        onnx_file_path,  # Where to save the ONNX model
        export_params=True,  # Store the trained parameter weights inside the model file
        opset_version=opset_version,  # The ONNX version to export the model to
        do_constant_folding=True,  # Whether to execute constant folding for optimization
        input_names=["input"],  # The model's input names
        output_names=["output"],  # The model's output names
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },  # Dynamic axes for variable length axes
    )

    print(
        f"Model has been successfully converted to ONNX format and saved to {onnx_file_path}"
    )


# Example usage:
# Assuming you have a PyTorch model and a sample input tensor
# model = YourPyTorchModel()
# input_tensor = torch.randn(1, 3, 224, 224)  # Example input tensor
# onnx_file_path = 'model.onnx'
# convert_pytorch_to_onnx(model, input_tensor, onnx_file_path)
