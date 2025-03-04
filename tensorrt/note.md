# TensorRT
https://github.com/NVIDIA/trt-samples-for-hackathon-cn

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
