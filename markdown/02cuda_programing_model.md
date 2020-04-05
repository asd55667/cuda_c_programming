# 第二章
## 编程模型
|SM
|:-:|-:
|应用程序|
|   V    |
|编程模型
|   V
|编译器/库
|   V
|操作系统
|   V
|芯片架构


- 算法与程序设计: domain level
- - 解析数据，解决问题，
- 编程阶段: logic level
- - 管理线程, 确保计算正确， #C并行编程 用pthreads或OpenMP
- 性能优化: hardware level
- - 理解线程如何映射至计算单元

### CUDA编程结构
[Unified Memory](https://zhuanlan.zhihu.com/p/82651065)
联合了gpu与cpu的内存空间，可以用同一个指针访问gpu和cpu上的数据

kenel编程模型跑在GPU上， 可以把kernel当作串行程序

CUDA编程模型是异构的，故可以与host-device通信重叠

> 一个CUDA程序有以下步骤:
- data: cpu -> gpu
- 调用kernel完成并行任务
- cpu <- gpu :data

cuda api 与 c std lib提供的api高度相似

> 函数签名:

- cudaMalloc
- - cudaError_t cudaMalloc (void\*\* devPtr, size_t size)


- cudaMemcpy
- - cudaError_t cudaMemcpy (void\* dst const void\* src, size_t count, cudaMemcpyKind kind)
- - - cudaMemcpyKind
- - - - cudaMemcpyHostToHost
- - - - cudaMemcpyHostToDevice
- - - - cudaMemcpyDeviceToHost
- - - - cudaMemcpyDeviceToDevice

cudaMemcpy这里有一个同步操作，需要等所有的blcoks拷贝完，除了launch一个kernel,每一个CUDA调用都会返回一个枚举型为cudaError_t的错误代码, 具体的, 当gpu内存成功分配返回cudaSuccess, 否则返回cudaErrorMemoryAllocation

char\* cudaGetErrorString(cudaError_t error) 类似 C strerror

> GPU内存分为两种

- 全局内存
- - 与CPU系统内存一致
- 共享内存
- - 类似CPU缓存， 而GPU共享内存可直接由kernel控制

CUDA编译器允许使用nvcc内嵌套的编译工具进行编译
-Xcompiler 声明 C编译器

### 线程管理
单个核函数launch的所有线程在共享同一块全局内存空间, grid下线程block的协作通过
- block-local synchronization
- blcok-local shared memory 
不同线程块下的线程不可协作

blockIdx, theadIdx变量为预初始化变量,可被kernel函数访问, CUDA runtime 会在kernel函数执行时将其分配至对应线程块中的线程

通常grid为两维， block为三维

dim3由host自行定义， 仅在host可见, device仅可见uint3


对给定数据，需决定grid于block的dim:
- 1. 分配block的dim
- 2. 根据block与数据大小计算grid的dim 

对block的dim, 你需要考虑:
- kernel的性能
- GPU资源的限制 如寄存器，共享内存



### Launch CUDA的kernel函数
kernel函数形如:
kernel_name<<<grid, block>>> (args);
- grid指block数
- blcok指的线程数
通过<<<>>>可配置布局


kernel的定义:
__gloval_ void kernel_name(args);
kernel返回须为空

__device__ & __host__可一起使用, 函数在两边都会编译

kernel的局限:
- 仅能访问设备内存
- 必须返回void
- 不支持变量参数
- 不支持静态变量
- 不支持函数指针
- 异步


kernel调用与host线程是异步的，在kernel被invoke之后， 控制立即交还给host
cudaError_t cudaDeviceSynchronize(void);强制host应用等待kernel完成

一些CUDA runtime API是隐式同步的:
cudaError_t cudaMemcpy(void \*dst, const void \*src, size_t count, cudaMemcpyKind kind);

调试时将kernel设置为<<<1,1>>>来验证计算结果


### 管理多GPU
> 查询GPU设备的所有信息

cudaError_t cudaGetDeviceProperties(cudaDeviceProp \*prop, int device);



