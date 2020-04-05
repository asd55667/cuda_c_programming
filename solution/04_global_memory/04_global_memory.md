### 全局内存

#### 层级内存的优势
|speed||Size
|:-:|:-:|:-:
|fastest||寄存器|smallest|
|\||缓存|\||
|v|主内存|v|
|slowest|磁盘内存|biggest




- Temporal locality (locality in time)
假设被引用内存地址在短时间内会再次被引用, 随着时间的推移会被引用的越来越少
- Spatial locality (locality in space)
假设被引用内存地址附近的地址可能被引用

gpu与cpu的层级内存相似, 不同在于CUDA编程模型给予开发者对其更多的显示控制


#### CUDA内存模型
在cpu中,L1与L2缓存为不可编程内存
gpu中, CUDA内存模型的可编程内存主要分为:
- 寄存器
- 共享内存
- Local内存
- Constant内存  ROM
- Texture内存   ROM
- 全局内存

不同内存空间拥有不同的作用域,生命周期,缓存行为.线程块由它自己的共享内存, 对块内所有的所有线程可见,持续到线程块被释放.线程亦可以访问全局内存,texture内存,Constant内存.

Texture内存提供不同的寻址方式,用以区分不同的数据布局.
全局,texture,constant随着应用程序而存在


#### 寄存器
- kernel中的无特定修饰符的自动变量通常存在寄存器中
- kernel中的数组仅在其引用的索引为常量,且该常量能在编译时被确定的情况下会被存到寄存器

寄存器变量对线程不可见,寄存器的生命周期与kernel相同

在kernel中使用更少的寄存器可以让SM拥有更多的线程块

当kernel使用的寄存器超出硬件限制, 寄存器会溢出到local内存,会影响性能,nvcc会启发式地最小化kernel的寄存器

手动设置
```
__global__ void 
__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor) 
kernel(...) {
 // your kernel body
}
```

kernel中可被分配到寄存器的变量但却分配不到的时候会被划到local内存中
- local数组引用的索引值不能再编译时被确定
- 会消耗大量寄存器内存的比较大的local数组,结构体
- 任何超出寄存器上限的变量

local内存这个概念是有些混淆的, 溢出到local内存的值实际上与全局内存是处在同一个物理位置.对算力在2.0以上的gpu,local内存的数据还会被缓存到每个SM的L1缓存以及每个设备的L2缓存上  


#### 共享内存
每个SM上的共享内存将被线程块划分

共享内存上的寻址须要进行块内同步

SM上的L1缓存与共享内存可以在运行时动态进行分配
```
cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache cacheConfig);
 ```

 #### Constant内存
constant变量在kernel外部定义, 在编译的空间范畴内对其所有的kernel可见

初始化
```
cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count);
 ```
constant内存适合用来广播, 当一个warp访问同一个地址时效率达到最高,
如相关系数等

#### Texture内存
texture内存是经过特定优化的全局内存, 适合具由2D空间局部特性的数据, 位于SM的只读缓存上


#### 全局内存
需要注意多个线程对全局内存的访问, 应避免不同线程块的线程并发对同一全局内存上的变量进行修改时

优化内存的transaction对性能是极为重要的.当warp进行加载存储操作,响应的transaciton数目能满足请求的两个因素为:
- warp内线程的内存地址分布
- 每个transaction内存地址的Alignment

#### GPU缓存
- L1
- L2
- 只读Constant
- 只读Texture

CUDA允许配置读取时的缓存
- 缓存到L1和L2
- 缓存到L2



#### Pinned内存
分配host内存默认是pageable类型

gpu没有pageable内存的控制权,host操作系统可能会移动该处的数据.
CUDA驱动先分配临时的page-locked或是pinned host内存
> cudaError_t cudaMallocHost(void **devPtr, size_t count);

1050ti
- memTransfer.cu
 GPU activities:   51.29%  13.314ms         1  13.314ms  13.314ms  13.314ms  [CUDA memcpy DtoH]
                   48.71%  12.644ms         1  12.644ms  12.644ms  12.644ms  [CUDA memcpy HtoD]

- pinnedMemTransfer.cu
 GPU activities:   54.39%  6.6974ms         1  6.6974ms  6.6974ms  6.6974ms  [CUDA memcpy DtoH]
                   45.61%  5.6161ms         1  5.6161ms  5.6161ms  5.6161ms  [CUDA memcpy HtoD]

使用pinnedMem会占用host的资源, 可以加速h2d以及d2h的速度,但加剧了alloc与free的时间

将批量小数据组合成一个大数据可以提升性能

#### Zero-Copy Memory
通常,host与device变量间不能相互访问.而zero-copy mem是两边均可访问
其优势在:
- gpu显存不足借用cpu内存
- 避免异构间显式的数据传输
- 提升PCIE传输速率

zero-copyMem实际上是映射到gpu地址空间的pinnedMem,在其之上的异构操作亦须同步
cudaError_t cudaHostAlloc(void **pHost, size_t count, unsigned int flags);
> flag
- cudaHostAllocDefault          与cudaMallocHost功能一致
- cudaHostAllocPortable         返回能被所有CUDAcontexts使用的pinnedMem
- cudaHostAllocWriteCombined    返回write-combined mem,适合被host写,dev读
- cudaHostAllocMapped           返回映射到dev地址空间的hostMem


cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)返回的指针可以供dev引用来访问映射的地址


#### Unified Virtual Addressing



