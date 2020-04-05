##CUDA执行模型

GPU架构有一组流处理器(SM, Stream multiProcessors)构成,

SM支持gpu上千个线程的并发执行

启动配置决定每个SM的block数目, SM的最大线程数通常大于block的最大线程数, 以在不同任务下更灵活

|-------- SM 层级结构 --------
|:-:|
|        指令缓存
|        warp scheduler x n  
|        dispatch unit
|        寄存器文件
|   核心, 加载/存储 单元, SFU
|        内连网络
|        共享内存/L1缓存
|        uniform缓存
            
SFU(Special func unit) 执行如sin, cos, sqrt, iterporlate

以SIMT(单指令多线程)架构管理并执行的一组32个线程为为一个warp,同一个warps中的thead在同一时刻执行同一指令,SM以warp为单位划分blcok

SIMT与SIMD的不同, SIMT允许同一个warp内的线程独立执行,
SIMT有但SIMD没有的三个特征:
- 每个线程都有独立的指令地址counter
- 每个线程都有独立的寄存器状态
- 每个线程都能有独立的执行路径

针对warp的边界(32的线程数)来优化工作量能更有效的利用GPU资源

共享内存分配给SM上的block, 寄存器分配给线程, 线程凭此通信

CUDA支持block内的同步

block内的warps是无序的, warps的并行数目受限于SM资源, 当一个warp挂起,SM会启动统一block内的其他warp, 由于SM在硬件层面对线程的划分, 会保存新开启warp的状态


### Profile-Driven 优化
Profiling通过:
- 内存或时间复杂度
- 特定指令的使用
- 函数调用的频率,间隔
来分析程序

对metric:
- 大多数Counters是针对每个SM,而不是整个GPU
- 每次的Counter不会一模一样, warp和block的scheduling顺序会不同
- 单次仅有几个counters, 一些counters互斥


### warps & block
warp是SM的基本执行单元, 一旦block被SM安排上了, block会被进一步划分为warp,
warp由连续的32个thread组成,并以SIMT方式执行

block被划分的warp数 = block的线程数/32 向上取整

如果block的大小不是32的倍数, 那么最后一个warp中会有线程inactive, 占用了SM资源却不参与计算

### warp divergence
同一个warp内执行不同的指令的情况称为warp divergence, 十分影响性能

要得到最佳性能, 应该保证warp内的线程是同一个执行路径

查看branch的Divergency情况
> $nvprof --metrics branch_efficiency simpleDivergency.cu

branch efficiency = 100 * ($branches - $divergent branches)/$branches

CUDA编译器会对代码进行优化，当条件声明内的指令低于一定阈值时 ,将branch指令较短的代码段替换为predicate指令, 

编译时不进行优化
> $nvcc -g -G -arch=sm_35 ...

获取event counters
> $nvprof --events branch,divergent_branch app

注：
- warp divergence仅在warp内的线程被划分到不同分支
- 不同的条件分支串行执行
- 将branch的颗粒度设置为warp的整数倍可避免divergence
- 不同的warp执行不同代码没有性能损耗


### 资源划分
一个warp的local execution context主要包括：
- 程序Counters
- 寄存器
- 共享内存

减少kernel消耗的寄存器数量， SM可以同时处理更多的warps， 
减少block消耗的共享内存，SM可以同时处理更多的blocks

当SM上的共享内存及寄存器连一个block也构建不了，kernel就无法执行

当一个block被分配了共享内存及寄存器， 那么该block就处于actvie状态，其包含的warps也是如此

active warp的状态可进一步可被划分为：
- Selected 		正在执行的warp
- Stalled 		未执行但做好了执行准备
- Eligibale 	未做好执行准备

eligiable 的两个必要条件:
- 有32个可执行的CUDA core
- 当前指令的所有参数已经就绪

需要保持大量的active warp来弥补warp stalling造成的延迟


### Latency Hiding
一个指令起止间时钟周期次数称为指令延迟，当所有的warp scheduler在每一个时钟周期均有eligible warp时，gpu利用率满载。

cpu旨在最小化同一时间一两个线程间的延迟，
gpu旨在处理大量并发，轻量线程时最大化吞吐量

当考虑指令延迟，延迟可分为
- 算术指令 算术操作起始至输出结果的时间 CPI 10-20
- 内存指令 加载或存储起止时间			CPI 400-800 全局内存访问


掩盖延迟所需的active warps
required warps = latency * throughput

> 算术延迟
带宽通常是理论峰值， throughput通常是实际值

throughput由每个SM每个周期的操作数决定

算术操作的并行条件与操作数/warp数挂钩

> 内存延迟
并行条件与每个周期的bytes数挂钩
memory throughput gigbytes/s

提升并行度：
- 每个线程拥有更多独立的指令/内存操作
- 更多并发的eligible thread

### 占用率
占用率 = activate warps / maximium warps

maximium warps:
> cudaError_t cudaGetDeviceProperties(struct cudaDevice \*devp, int dev);

通过变量maxThreadPerMultiProcessor查看每个SM的最大线程数
maxmium warps = maxThreadPerMultiProcessor / 32

> grid and block 尺寸guidelines
- block中的线程数为warpSize的整数倍
- 避免block过小， 至少128或256个线程
- 根据kernel的资源要求来适当调整block
- 控制block的个数远大于GPU中SM的个数
- 通过实验来发现最佳配置方案

###  同步
CUDA中的同步分为两个层级
- 系统层	host和device间
- block层	block中所有进程

cudaDeviceSynchronize();可能会返回先前的cuda的异步操作

在kernel中标记同步
> __device__ void __syncthreads(void);
__syncthreads调用后，所有线程会同时到达同步位置，位置前的所有线程访问的内存将公开给位置后的线程

> --metrics 参数

- gld_throughput	内存读取速率
- gld_efficiency	ratio of requested global load throughput to required global load throughput
- achieved_occupancy 占用率
- inst_per_warp     warp指令数
> metrics and performance
- 绝大多数情况， 单一指标不能准确的衡量最秀性能
- 总体性能取决于kernel的实现


### 规避分支的Divergence
> reduceInteger.cu
- reduceNeighbored 的条件语句*if(tidx%(2\*i)==0)*仅在索引为偶数的线程上执行inplace操作
- reduceNeighboredLess 将条件语句映射至表达式*j = 2 * i * tidx*后, 缓解了部分的warp divergence
- reduceInterleaved 每一个循环会减少占用的资源， divergence程度与reduceNeighboredLess一致

> Unrolling loops

```
for (int i = 0; i < 100; i++) 
    a[i] = b[i] + c[i];

for (int i = 0; i < 100; i += 2) {
 a[i] = b[i] + c[i];
 a[i+1] = b[i+1] + c[i+1];
}
```
在循环体内增加重复指令,减少循环的次数,以减少分支的频率和循环指令来优化循环执行

- unroll warps
```
if (tid < 32) {
 volatile int *vmem = idata;
 vmem[tid] += vmem[tid + 32];
 vmem[tid] += vmem[tid + 16];
 vmem[tid] += vmem[tid + 8];
 vmem[tid] += vmem[tid + 4];
 vmem[tid] += vmem[tid + 2];
 vmem[tid] += vmem[tid + 1];
}
```
规避了执行循环所需控制以及线程同步逻辑

- complete unroll
```
if (iBlockSize>=1024 && tid < 512) idata[tid] += idata[tid + 512];
 __syncthreads();
if (iBlockSize>=512 && tid < 256) idata[tid] += idata[tid + 256];
 __syncthreads();
    ……
```
nvcc会自动移除false的分支, 前提是必须以swtich-case的形式

> 1050ti
```
4.32%  11.547ms  reduceNeighbored(int*, int*, unsigned int)
2.42%  6.4689ms  reduceNeighboredLess(int*, int*, unsigned int)
2.17%  5.7893ms  reduceInterleaved(int*, int*, unsigned int)
1.29%  3.4456ms  reduceUnrolling2(int*, int*, unsigned int)
0.69%  1.8551ms  reduceUnrolling4(int*, int*, unsigned int)
0.42%  1.1108ms  reduceUnrolling8(int*, int*, unsigned int)
0.35%  946.47us  reduceUnrollWarps8(int*, int*, unsigned int)
0.35%  940.16us  reduceCompleteUnrollWarps8(int*, int*, unsigned int)
0.35%  929.09us  reduceCompleteUnroll<unsigned int=512>(int*, int*, unsigned int)
```


### 动态并行
CUDA允许kernel在GPU端被创建,同步. 直接在GPU端创建任务能减少与CPU控制及数据的交替

### Nested execution
- 父线程显示同步后才能保证子线程开始执行
- 父线程所有的全局内存操作优先于子线程，可保证其对子线程可见
- 子线程的所有的内存操作在父线程同步后才对父线程可见

### Nested Reduction
递归由block的第一个thread压入栈
大量的kernel调用及块内同步调用亦会影响主kernel性能

编写高效的nested kernel，需要注意gpu代码的运行时间，子节点的启动策略，父子节点的同步，递归深度