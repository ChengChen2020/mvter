# 4.22

生成transformed数据集

两个问题

![截屏2021-04-22 下午10.14.45](/Users/julius/Library/Application Support/typora-user-images/截屏2021-04-22 下午10.14.45.png)

1. 有些类别的off文件通过安装的`off2obj`没法转换，

   `bathtub`(156),`desk`(286),`dresser`(286),`monitor`(565),`night_stand`(286),`sofa`(780),`table`(492),共

   156+286+286+565+286+780+492=2851个样本没rotate成功

   9449个成功

   明天进一步处理

2. blender 里面可以直接指示欧拉角进行rotate，$(\alpha, \beta, \gamma)$, 所以好像可以只存三个轴旋转的角度，当作$t$



# 4.23

构建网络

1. 每个view是$3\times224\times224$, 我把12个view堆成$36\times224\times224$送进网络，再切片分别提取特征
2. transformation decoding module, i.e., $D(\cdot)$, 设置成一个全连接层，输出3个值对应欧拉角
3. task module, i.e., $T(\cdot)$, 设置成一个全连接层，输出（40个）类别的概率分布

不知道这么设计是不是可以

准备用转换成功的9449对sample，33个类别，先试试看看



# 4.24

Dataloader

1. 开始打算先把raw数据保存成hdf5文件，然后MB级的数据成了GB级（可能是哪里出了差错），遂还是直接dataloader

2. 共9449个sample，7589个train，1860个test，train/test比约为2:8

3. 目前看起来还正常

   ![截屏2021-04-24 下午10.26.21](/Users/julius/Library/Application Support/typora-user-images/截屏2021-04-24 下午10.26.21.png)

   

# 4.25

把东西运到了server，在`~/chengc/mvter/`

dataloader和model基本集成起来了

还差一个`trainer`



# 4.28

I'm trying to reproduce the results of MVTER(GVCNN) in paper, but the initial rounds acc is around 93 which is less than 97 in paper. The reason could be errors in my implementation. I'll observe results of further epochs while examining my implememtation. 



# 4.29

The vesta server is not free and I cannot get enough memory to train with a suitable batch size. I've passed the quiz for HPC, hopefully I can tune the hyperparameters on Dalma soon.



# 4.30

While waiting for access to Dalma, I begin reading the paper about Robust Contrastive Learning(RoCL). Actually I'm doing a course project about contrastive self-supervised learning, i.e. SimCLR related. I now have a preliminary understanding of integrating contrastive learning with adversarial learning.