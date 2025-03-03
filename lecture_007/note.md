# Advanced Quantization
## Origin Slides
### Dynamic Quantization
+ Recalculates quantization parameters for each
sample
  - Insensitive to non-stationary distributions
  - Sensitive to frequent outliers
+ Floating point activations
  - Drop in replacement for non quantized op
  - Can be slower than techniques which allow for a
series of quantized ops without dequantizing

See dynamic_quantization.py for per token + per channel quantization. A coarse-grained version of per-channel quantization is to use different quantization steps for different channel groups, called group-wise quantization.

### Weight Only Quantization Int8/Int4
1. **GPU occupancy**
2. **It does more work than base matmul**
This can be solved by torch.compile

### Static Quantization
Do calibration for activation quantization.
+ Calculates best quantization parameters over calibration set. 
  - More sensitive to non-stationary distributions.
  - Less sensitive to frequent outliers.
+ Integer activations
  - Best if have a sequence of quantizable ops.

### Code Pointers
+ Quantization API https://github.com/pytorch-labs/ao
+ SAM https://github.com/pytorch-labs/segment-anything-fast
+ GPTQ https://github.com/pytorch-labs/gpt-fast

## Other Materials
Papers
+ [A Survey on Model Compression for Large Language Models](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00704/125482)
+ [GPTQ](https://arxiv.org/pdf/2210.17323)
+ [AWQ](https://arxiv.org/pdf/2306.00978)
+ [SmoothQuant](https://arxiv.org/pdf/2211.10438)
+ [KVQuant](https://proceedings.neurips.cc/paper_files/paper/2024/file/028fcbcf85435d39a40c4d61b42c99a4-Paper-Conference.pdf)
+ [LLMQAT](https://arxiv.org/pdf/2305.17888)

Links
+ https://zhuanlan.zhihu.com/p/703928680


## Quantization
### Quantization Aware Training (QAT)

### Post Training Quantization (PTQ)
#### Weight Only Quantization
#### Weight Activation Quantization
#### KV Cache Quantization


## Pruning
## Knowledge Distillation
## Low Rank Factorization


## Other Questions
### The difficulties of activation quantization
The quantization of activations is generally considered more difficult than the quantization of weights in neural networks for several reasons:

1. **Dynamic range variability**:
   - **Weights**: The weights of a neural network are relatively static once the network is trained. Their values typically fall within a certain range that can be analyzed and characterized during the training process. For example, in a convolutional neural network (CNN), the weights of the convolutional filters have a relatively stable distribution of values. Once determined during training, they do not change during the forward pass of inference, and it is possible to find an appropriate quantization scheme (such as fixed-point quantization) that can represent them with a lower precision while maintaining acceptable accuracy.
   - **Activations**: In contrast, the activations of neurons in a neural network can have a highly variable dynamic range during the forward pass of inference. The output of an activation function in a layer can vary widely depending on the input data. For instance, in an image recognition task, different input images can cause the activations of neurons in a convolutional layer to have very different magnitudes. This makes it challenging to find a single quantization scheme that can effectively represent all possible activation values without significant loss of information.

2. **Distribution complexity**:
   - **Weights**: The distribution of weights often has certain regularities. For example, many weights in neural networks follow a Gaussian or near-Gaussian distribution. This allows for the application of quantization techniques that can exploit these statistical properties. For example, quantization methods can be designed to group the weight values based on their distribution characteristics and represent them with fewer bits.
   - **Activations**: The distribution of activations is usually more complex and less predictable. Activations can have multimodal distributions, outliers, and non-stationary characteristics. For example, in a neural network processing natural language data, the activations of neurons in a recurrent layer can have a wide variety of values depending on the structure and content of the input sentences. These complex distributions make it difficult to design quantization algorithms that can accurately represent the activations while reducing the precision.

3. **Sensitivity to errors**:
   - **Weights**: Neural networks are often somewhat robust to small errors introduced by weight quantization. Since the weights are learned during the training process, a certain level of quantization error can be tolerated without significant degradation of the overall performance. The training process can even be adjusted to optimize the network for quantized weights, such as through techniques like quantization-aware training.
   - **Activations**: Activations are more sensitive to quantization errors. Small inaccuracies in representing the activation values can have a cascading effect on the subsequent layers of the neural network. For example, if the activation values in an early layer of a CNN are quantized inaccurately, it can lead to incorrect feature representations being passed on to the later layers, which can significantly degrade the performance of the network in tasks like object detection or image classification.

4. **Lack of global information**:
   - **Weights**: The weights of a neural network are global parameters of the network, and it is possible to analyze their overall distribution and characteristics across the entire network. This global view allows for the design of quantization schemes that can optimize the representation of weights across all layers.
   - **Activations**: Activations are generated on a per-input basis, and there is no straightforward way to obtain a global view of all possible activation values. Each input sample can generate a different set of activation values, making it difficult to design a single quantization scheme that is optimal for all possible inputs. This lack of global information about activations makes their quantization more challenging compared to weights.

In summary, the combination of dynamic range variability, complex distributions, sensitivity to errors, and lack of global information makes the quantization of activations more difficult than the quantization of weights in neural networks.  

### Quantization of ReLU (Rectified Linear Unit)
The ReLU function is defined as \(f(x) = \max(0, x)\). When quantizing ReLU activations, the following general steps can be involved:

**Step 1: Determine the quantization range**
Since ReLU outputs non-negative values, the quantization range will typically start from 0. To find the upper bound of the range, one common approach is to analyze the distribution of activation values during a representative set of input examples (e.g., a validation dataset). For example, you might calculate the maximum activation value over a large number of samples or use a statistical measure like the 99th percentile of the activation values to set an appropriate upper limit.

**Step 2: Select a quantization method**
- **Uniform quantization**: In uniform quantization, the range determined in the previous step is divided into a fixed number of intervals. For example, if you want to use 8-bit quantization for the ReLU activations and you've determined the range to be from 0 to \(V_{max}\), you divide the interval \([0, V_{max}]\) into 256 (since \(2^8 = 256\)) equal sub-intervals. Each activation value \(x\) is then mapped to the closest integer value within this quantized range. Mathematically, the quantized value \(q\) of an activation \(x\) can be calculated as \(q = \text{round}(\frac{x}{s})\), where \(s=\frac{V_{max}}{255}\) is the quantization step size.
- **Non-uniform quantization**: In some cases, non-uniform quantization can be more effective as the distribution of ReLU activations may not be evenly spread. For example, if there are many small activation values and fewer large ones, non-uniform quantization can allocate more bits to represent the smaller values accurately and fewer bits for the larger values. One common approach is logarithmic quantization, where the quantization intervals are spaced logarithmically.

**Step 3: Dequantization during inference**
During the forward pass of the neural network, the quantized values need to be dequantized back to the original floating-point representation for use in subsequent layers. This is done by multiplying the quantized integer value \(q\) by the quantization step size \(s\) used during quantization (\(x_{dequantized}=q\times s\)).


### 2. GLU (Gated Linear Unit)
The GLU activation function is defined as \(GLU(x) = x_1 \odot \sigma(x_2)\), where \(x = [x_1, x_2]\) is the input vector, \(\odot\) is the element-wise multiplication operation, and \(\sigma\) is the sigmoid function.

**Step 1: Quantize the components**
- For the input vector \(x\), split it into \(x_1\) and \(x_2\) as per the GLU definition. Each of these components can be quantized separately using a suitable quantization method (e.g., uniform or non-uniform quantization as described for ReLU).
- For the sigmoid part \(\sigma(x_2)\), since the sigmoid function outputs values in the range \([0, 1]\), a quantization scheme can be designed to represent these values with a lower precision. For example, you can use fixed-point quantization where the range \([0, 1]\) is mapped to a set of integer values within a certain range (e.g., for 8-bit quantization, mapping to values from 0 to 255 and then scaling appropriately back to the \([0, 1]\) range during dequantization).
- After quantizing \(x_1\) and \(\sigma(x_2)\), perform the element-wise multiplication operation in the quantized domain.

**Step 2: Adjust for potential overflow and underflow**
Since the multiplication operation in GLU can result in values that may be outside the expected range of the quantized representation, special handling may be required to prevent overflow or underflow. This could involve clipping the resulting values to an appropriate range or using more advanced quantization techniques that account for such arithmetic operations.

**Step 3: Dequantization for subsequent layers**
Similar to ReLU, the final quantized output of the GLU operation needs to be dequantized before being passed to the next layer of the neural network.


### 3. SwiGLU (Simplified Gated Linear Unit)
SwiGLU is a variant of GLU and is defined as \(SwiGLU(x) = x_1 \odot \text{gelu}(x_2)\), where \(\text{gelu}\) is the Gaussian Error Linear Unit.

**Step 1: Quantize the input components**
- Split the input \(x\) into \(x_1\) and \(x_2\) as in GLU. Quantize \(x_1\) using an appropriate quantization method.
- For the \(\text{gelu}(x_2)\) part, the \(\text{gelu}\) function has a more complex behavior compared to the sigmoid in GLU. One approach is to approximate the \(\text{gelu}\) function with a simpler piecewise linear or polynomial function that can be more easily quantized. Once the approximation is obtained, the output of this approximated function (which is in the range of \(\text{gelu}\) values) can be quantized using a suitable quantization scheme, such as uniform quantization over the determined range of the approximated function's output.
- Then perform the element-wise multiplication between the quantized \(x_1\) and the quantized \(\text{gelu}(x_2)\) values.

**Step 2: Error handling and optimization**
Due to the approximation of the \(\text{gelu}\) function and the quantization process, there will be some error introduced. Techniques such as quantization-aware training can be used to fine-tune the network during training to minimize the impact of these errors on the overall performance of the neural network.

**Step 3: Dequantization**
Finally, dequantize the output of the SwiGLU operation to the floating-point representation for use in the subsequent layers of the network. 



In all cases, the goal of quantization is to represent the activation values with a lower precision while minimizing the degradation in the performance of the neural network. 
