# TensorRT
https://github.com/NVIDIA/trt-samples-for-hackathon-cn

## 基本流程
### 构建
+ 建立logger
+ 建立builder
  + buidler config
    + workspace
    + int8_calibrator
    + flag 设置FP16、INT8、TF32 refit模式 手工数据类型限制等
    + add_optimization_profile 添加dynamic shape输入配置器
  + builder.create_network()
  + builder.create_optimization_profile()
+ 创建network
  + 常用参数 1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) 使用explicit batch
  + 常用方法 add_input add_convolution_nd 添加网络层 mark_output
  + 成员变量 network.name network.num_layers network.num_inputs network.num_outputs
  + network.has_implicit_batch_dimension() network.has_explicit_precision()
  + Explicit batch比implicit batch多一维 ONNX导入默认前者 推荐前者 额外支持
    + batch norm 
    + reshape transpose reduce over batch
    + dynamic shape
    + loop结构
    + layer高级用法 如 shuffle_layer.set_input
  + Dynamic shape需要explicit batch
    +  需要optimization profile帮助网络优化 builder.create_optimization_profile() profile.set_shape() config.set_optimization_profile
    +  需要context.set_input_shape绑定实际输入shape
  + layer成员 name type precision get_input(i) get_output(i) network.get_layer()
  + tensor成员 name shape dtype
  + 一般FP32相对误差1e-6 FP16相对误差1e-3
  + FP16模式
    + config.flags = 1 << int(tensorrt.BuilderFlag.FP16)
    + 比FP32建立时间更长 需要选择kernel和插入reformat节点 (timeline中会有reformat节点 如nchwToNchw)
    + 部分层误差较大 需要polygraphy 强制该层FP32计算 config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTs) layer.precision = trt.float32
  + INT8模式 PTQ
    + 需要校准集合
    + 实现calibrator config.set_flag(trt.BuilderFlag.INT8) config.int8_calibrator = calibrator
  + INT8模式 QAT
    + config.set_flag(trt.BuilderFlag.INT8)
    + 在pytorch中插入 quantize 和 dequantize 节点
+ 生成serialized network
  + serialized_network = builder.build_serialized_network()
  + 需要环境完全一致
  + engine不同生成不保证一样 可以algorithm selector和timing cache多次生成一样的engine
  
### 运行阶段
+ 建立engine
  + engine = trt.Runtime(logger).deserialize_cuda_engine(serialized_network)
  + engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
  + engine.num_layers
  + engine.get_tensor_name(i)
  + engine.get_tensor_dtype(name\[i\])
  + engine.get_tensor_shape(name\[i\])
  + engine.get_tensor_mode(name\[i\]) 输入输出
+ 创建context
  + engine.create_execution_context()
  + 绑定动态输入 context.set_input_shape(name, shape)
  + context.execute_async_v3(stream)
+ buffer准备与拷贝
+ 推理
+ buffer拷贝
+ 善后工作

### ONNX 
+ pytorch .pt -> ONNX -> ONNX simplifier -> TensorRT
+ 需要parser 不支持节点需要修改模型或者ONNX 实现plugin 或者修改TensorRT

## Tools
### trtexec
功能
+ ONNX生成TensorRT引擎并序列化为Plan文件
+ 查看ONNX文件或者Plan文件的网络逐层信息
+ 测试在给定或者随机输入下的网络性能
构建期常用选项
+ --onnx
+ --minShapes=x:0:1x1 --optShapes=x:0:10x10 --maxShapes=x:0:20x20
+ --memPoolSize=workspace:1024MiB
+ --fp16 --int8 --noTF32 --best --sparsity= 
+ --saveEngine=engine.plan
+ --skipInference
+ --verbose
+ --preview=
+ --builderOptimizationLevel=
+ --timingCacheFile
+ --profilingVerbose
+ --dumpLayerInfo --exportLayerInfo=layer.txt
运行期常用选项
+ --loadEngine=engine.plan
+ --shapes=x:0:1x1
+ --warmUp=1000
+ --duration=1000
+ --iterations=1000
+ --useCudaGraph
+ --noDataTransfers
+ --streams=2
+ --verbose
+ --dumpProfile --exportProfile=profile.txt
重要指标
+ 吞吐量
+ 延迟统计及解释

### Netron
查看网络结构

### onnx-graphsurgeon
需要修改网络的情形
+ 冗余节点
+ 阻碍TensorRT融合的节点组合
  + 如conv和relu中增加了squeeze操作
+ 可以手工优化的节点
  + LayerNorm

Node 注意避免修改上下游

Variable 

常量折叠，可以用onnx-simplifier或者polygraphy

### polygraphy
功能
+ 使用多种后端推理
+ 比较不同后端精度
+ 由模型文件生成TensorRT引擎并序列化
+ 查看网络逐层信息
+ 修改ONNX模型，提取子图，计算图化简
+ 分析ONNX转TensorRT失败原因，将原计算图中可以或不可以转TensorRT的子图分割保存
+ 隔离TensorRT中的错误tactic

RUN模式选项
+ --onnx-outputs mark all
+ --trt-outputs mark all  
+ --model-type 指定后端
+ --input-shapes "x:[4,1]" "y:[4,1]"

INSPECT模式功能
+ 判断onnx是否被TensorRT支持，切割支持和不支持的部分
+ 导出逐层详细信息

surgeon模式功能
+ 优化计算图

其他模式
+ convert模式，类似run模式
+ debug模式，检查模型转TensorRT错误并分离可运行的最大子图
+ data模式，调整分析运行模型的输入输出、权重
+ template模式，用于生成polygraphy的python脚本，使用脚本对模型调整
  
### nsight systems
+ 命令行 nsys profile XXX 获得.qdrep和.qdrep-nsys文件
+ 打开nsys-ui 将文件拖入观察timeline

建议
+ 只计量运行阶段
+ 构建时打开模型profiling获得layer更多信息，builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
+ 可以搭配nsys或自己脚本使用
+ nsys不能向前兼容

调试
+ 准备阶段，GPU计算阶段，结束阶段
+ GPU timeline从CUDA HW行展开
+ 从all stream和TensorRT show in event
+ launch bound需要cuda graph解决（CPU调用和GPU执行等待时间过长）

## Plugin
功能
+ 以so形式插入网络实现算子
+ 实现trt不原生支持的结构
+ 提高算子性能
+ 手动合并不能自动融合的层

限制条件
+ 需要写CUDA kernel，保证精度和性能
+ 无法和其他layer fusing
+ 可能需要reformat节点增加开销

建议
+ 优先尝试原生layer组合
+ 尝试自带plugin
+ 自己干

### 实现步骤
+ 继承IPluginV2DynamicExt类
  + plugin类型
    + IPluginV2 单一input output
    + IPluginV2Ext 单一input 混合output
    + IPluginV2IOExt 混合input 混合output implicit batch
    + IPluginV2DynamicExt 混合input 混合output dynamic shape
  + 构造函数 接受plugincreator初始化
  + getOutputDimensions 报告输出张量 不一定是构建期常量 允许表达式计算 outputIndex只索引张量
  + supportFormatCombination 支持多种dtype layout 输入输出组合
    + trt深度优先遍历 pos pos是输入输出张量索引 返回是否支持 dtype layout 当前组合
    + FP16 模式尝试kHalf 
    + INT8 模式尝试kINT8 如果需要支持calibration 需要fp32实现 否则手工指定输入输出张量dynamic range 内部张量dynamic range也需要指定
    + 在确保性能时 实现多种组合来消除format
  + configurePlugin
    + 推理前调用
    + dynamic shape模式 输入数据shape改变调用
    + 构建期 in out张量形状有 -1 运行期为真实形状
  + getWorkspaceSize
    + 报告中间计算的存储空间
    + 由trt管理参与显存优化
  + enqueue
    + 根据输入 shape layout 选择调用CUDA kernel
    + 不可以使用 cudamalloc 需要在workspace申请
  + initialize engine创建调用 初始化plugin
  + terminate engine销毁调用 释放initialize资源
  + clone 创建多个context 与源对象共享本engine资源
  + attachToContext 申请使用context独占的cudnn和cublas资源
  + detachFromContext 释放attachToContext申请的cudnn和cublas资源
  + destroy context或plugin销毁调用
  + 序列化 plugin负责
    + getSerializationSize 报告所需空间大小 byte
    + serialize 将plugin序列化到给定buffer中
  + 反序列化 creator负责
    + deserializePlugin 从buffer中传给plugin构造函数
    + plugin构造函数 完成读取数据和plugin构造
  + Version Namespace
    + trt将plugin name type version Namespace写入engine
    + 通常不修改
+ 继承IPluginCreator类
  + create_plugin 构造plugin
  + 注册plugincreator REGISTER_TENSORRT_PLUGIN(***PluginCreator)
+ 实现CUDA kernel
+ 编译so
+ 加载使用plugin
  + 构建期
    + trt向plugin传输参数权重
    + plugin向trt传输张量数量 shape dtype layout workspace
    + trt尝试允许组合选择性能最佳的输入输出组合 可能在plugin前后插入reformat节点
    + plugin不参加节点融合
  + 运行期
    + trt向plugin传输输入输出张量地址 workspace地址 stream
  + 加载so
  + 从registry中查找plugin
  + 通过creator创建plugin
  + 将plugin插入网络或者parser自动识别ONNX中的plugin

### plugin parser的结合使用
基本步骤
+ netron分析onnx需要替换的模块
+ onnx-surgeon替换新节点
+ 实现plugin
+ trt加载修改后的plugin和onnx
+ 对比加载前后精度和性能

### 优化范例
+ 整合零散算子 memory bound的操作 如多个输入预处理
+ 优化self attention等
+ 使用FastTransformer等高度优化的cuda kernel
