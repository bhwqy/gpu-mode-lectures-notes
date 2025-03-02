import numpy as np

def compute_scales(tensor):
    per_channel_max = np.max(np.abs(tensor), axis=(0, 2))
    per_channel_scale = per_channel_max / 127.0  # Assuming int8 range
    per_token_max = np.max(np.abs(tensor), axis=1)
    per_token_scale = per_token_max / 127.0  # Assuming int8 range
    return per_channel_scale, per_token_scale

def quantize_tensor(tensor, per_channel_scale, per_token_scale):
    batch_size, num_channels, sequence_length = tensor.shape
    quantized_tensor = np.zeros_like(tensor, dtype=np.int8)
    for c in range(num_channels):
        quantized_tensor[:, c, :] = (tensor[:, c, :] / per_channel_scale[c]).astype(np.int8)
    for b in range(batch_size):
        for t in range(sequence_length):
            quantized_tensor[b, :, t] = (quantized_tensor[b, :, t] / per_token_scale[b, t]).astype(np.int8)
    return quantized_tensor

def dequantize_tensor(quantized_tensor, per_channel_scale, per_token_scale):
    batch_size, num_channels, sequence_length = quantized_tensor.shape
    dequantized_tensor = np.zeros_like(quantized_tensor, dtype=np.float32)
    for b in range(batch_size):
        for t in range(sequence_length):
            dequantized_tensor[b, :, t] = quantized_tensor[b, :, t] * per_token_scale[b, t]
    for c in range(num_channels):
        dequantized_tensor[:, c, :] *= per_channel_scale[c]
    
    return dequantized_tensor

if __name__ == "__main__":
    batch_size = 4
    num_channels = 8
    sequence_length = 16
    tensor = np.random.randn(batch_size, num_channels, sequence_length)
    print("Original Tensor:\n", tensor)

    per_channel_scale, per_token_scale = compute_scales(tensor)
    print("Per-Channel Scale:\n", per_channel_scale)
    print("Per-Token Scale:\n", per_token_scale)

    quantized_tensor = quantize_tensor(tensor, per_channel_scale, per_token_scale)
    print("\nQuantized Tensor:\n", quantized_tensor)

    dequantized_tensor = dequantize_tensor(quantized_tensor, per_channel_scale, per_token_scale)    
    print("\nDequantized Tensor:\n", dequantized_tensor)

