from numpy import ndarray, exp, zeros

class LinearKernel:

    # 线性核函数（Linear kernel function，核函数值 K(x, z) = x @ y，显然它们都是二维的

    def __call__(self, X__: ndarray, Y__: ndarray) -> ndarray:
        K__ = zeros([len(X__), len(Y__)])  # 核函数值矩阵
        for n, x_ in enumerate(X__):
            K__[n] = Y__ @ x_
        return K__


class PolyKernel:

    # 多项式核函数（Polynomial kernel function），核函数值 K(x, z) = (γ * x @ y + r)**d，d、γ、r为超参数

    def __init__(self,
                 d: int = 2,
                 r: float = 1.,
                 γ: float = 1.,
                 ):
        assert type(d) == int and d >= 1, "多项式核函数的指数d应为正整数，K(x, y) = (γ * x @ y + r)**d"
        assert γ > 0, "多项式核函数的超参数γ应为正数，K(x, y) = (γ * x @ y + r)**d"
        self.d = d  # 指数d
        self.r = r  # 参数r
        self.γ = γ  # 参数γ
        self.linearKernel = LinearKernel()  # 实例化线性核函数

    def __call__(self, X__: ndarray, Y__: ndarray) -> ndarray:
        K__ = self.linearKernel(X__, Y__)  # 线性核函数的输出
        K__ = (self.γ * K__ + self.r) ** self.d  # 多项式核函数的输出
        return K__


class RBFKernel:
    """
    高斯核函数（Gaussian kernel function），也称径向基函数（Radial Basis Function）
    核函数值 K(x, y) = exp(-γ * sum((x - y)**2))
    其中，γ为超参数
    """

    def __init__(self, γ: float = 1.):
        assert γ > 0, "RBF核函数的超参数γ应为正数"
        self.γ = γ  # 参数γ

    def __call__(self, X__: ndarray, Y__: ndarray) -> ndarray:
        D2__ = (X__ ** 2).sum(axis=1, keepdims=True) + (Y__ ** 2).sum(axis=1) - 2 * X__ @ Y__.T  # 距离平方矩阵
        return exp(-self.γ * D2__)


class tlnnKernel:
    """
    两层神经网络two-layer neural network(tlnn)核函数K(x,y)=tanh(γ(xy)+r)
    其中，γ,r为超参数
    """

    def __init__(self,
                 γ: float = 1.,
                 r: float = 1.,
                 ):
        self.γ = γ  # 参数γ
        self.r = r  # 参数r

    def __call__(self, X__: ndarray, Y__: ndarray) -> ndarray:
        K__ = zeros([len(X__), len(Y__)])  # 核函数值矩阵，X__,Y__都为二维，故为方便直接定义二维
        for n, x_ in enumerate(X__):
            K__[n] = np.tanh(self.γ * Y__ @ x_ + self.r)
        return K__

# 支持向量机分类
from random import choice, random
from numpy import array, ones, zeros, ndarray, where, isnan, inf, intersect1d, maximum

class SupportVectorMachineClassification:
    """支持向量机分类"""
    def __init__(self,
            C: float = 1.,               # 超参数：惩罚参数
            kernel: str = 'linear',      # 核函数：可选 线性核'linear'/高斯核'rbf'/多项式核'poly/二层神经网络'tlnn'
            solver: str = 'SMO',         # 求解算法：可选 序列最小优化算法'SMO'
            maxIterations: int = 50000,  # 最大迭代次数
            tol: float = 1e-3,  # 迭代停止的收敛精度（指SMO求解算法当中最大违反KKT条件的程度）
            d: int = 2,         # 超参数：多项式核函数的指数
            γ: float = 1.,      # 超参数：高斯核函数、多项式核函数的参数
            r: float = 1.,      # 超参数：多项式核函数的参数
            ):
        assert C>0, '惩罚参数C应大于0'
        assert type(maxIterations)==int and maxIterations>0, '最大迭代次数maxIterations应为正整数'
        assert tol>0, '收敛精度tol应大于0'
        assert type(d)==int and d>=1, '多项式核函数的指数d应为不小于1的正整数'
        assert γ>0, '高斯核函数、多项式核函数的参数γ应大于0'
        self.C = C                    # 超参数：惩罚参数
        self.kernel = kernel.lower()  # 核函数：可选 线性核'linear'/高斯核'rbf'/多项式核'poly'/sigmoid核函数'tlnn'
        self.solver = solver.lower()  # 求解算法：序列最小优化'SMO''
        self.maxIterations = maxIterations  # 最大迭代次数
        self.tol = tol                # 迭代停止的收敛精度（指SMO求解算法当中最大违反KKT条件的程度）
        self.d = d      # 超参数：多项式核函数的指数
        self.γ = γ      # 超参数：高斯核函数、多项式核函数的参数、二层神经网络核函数的参数
        self.r = r      # 超参数：多项式核函数的参数、二层神经网络核函数的参数
        self.M = None   # 输入特征向量的维数
        self.w_ = None  # M维向量：权重向量
        self.b = None   # 偏置
        self.α_ = None  # N维向量：所有N个训练样本的拉格朗日乘子
        self.supportVectors__ = None  # 矩阵：所有支持向量
        self.αSV_ = None     # 向量：所有支持向量对应的拉格朗日乘子α
        self.ySV_ = None     # 向量：所有支持向量对应的标签
        self.minimizedObjectiveValues_ = None  # 列表：SMO求解算法，对偶问题的最小化目标函数值
        """选择核函数"""
        if self.kernel=='linear':
            self.kernelFunction = LinearKernel()
            print('使用线性核函数')
        elif self.kernel=='poly':
            self.kernelFunction = PolyKernel(d=self.d, γ=self.γ, r=self.r)
            print('使用多项式核函数')
        elif self.kernel=='rbf':
            self.kernelFunction = RBFKernel(γ=self.γ)
            print('使用高斯核函数')
        elif self.kernel=='tlnn':
            self.kernelFunction = tlnnKernel(γ=self.γ, r=self.r)
            print('使用二层神经网络核函数')
        else:
            raise ValueError(f"未定义核函数'{kernel}'")
        """选择求解算法"""
        if self.solver=='smo':
            print('使用SMO算法求解')

    def fit(self, X__: ndarray, y_: ndarray):
        """训练：使用SMO算法进行训练"""
        self.M = X__.shape[1]    # 输入特征向量的维数
        if self.solver=='smo':   # 使用SMO算法训练
            self.SMO(X__, y_)
        return self

    def SMO(self, X__, y_):
        """使用SMO（Sequential Minimal Optimization）算法求解对偶问题，最小化目标函数"""
        C = self.C      # 读取：惩罚参数
        N = len(X__)    # 训练样本数量
        K__ = self.kernelFunction(X__, X__)  # N×N矩阵：核函数矩阵
        α_ = zeros(N)   # N维向量：初始化N个拉格朗日乘子
        b = 0.          # 初始化偏置
        self.minimizedObjectiveValues_ = []  # 列表：记录历次迭代的目标函数值
        minimizedObjectiveValue = 0.5*α_ @ (y_*K__*y_.reshape(-1, 1)) @ α_ - α_.sum()  # 初始化：目标函数值
        for t in range(1, self.maxIterations + 1):
            indexSV_ = where(α_>0)[0]                   # 数组索引：满足α>0的支持向量
            indexNonBound_ = where((0<α_) & (α_<C))[0]  # 数组索引：满足0<α<C的支持向量，满足0<α<C的α称为非边界α（Non-bound）
            self.minimizedObjectiveValues_.append(minimizedObjectiveValue)   # 记录当前目标函数值
            """检验所有N个样本是否满足KKT条件，并计算各样本违反KKT条件的程度"""
            g_ = (α_[indexSV_]*y_[indexSV_]) @ K__[indexSV_, :] + b  # N维向量：g(xi)值，i=1~N
            E_ = g_ - y_                                       # N维向量：E值，Ei=g(xi)-yi，i=1~N
            yg_ = y_*g_                                        # N维向量：yi*g(xi)，i=1~N
            violateKKT_ = abs(1 - yg_)                         # N维向量：开始计算“违反KKT条件的程度”
            violateKKT_[(α_==0) & (yg_>=1)] = 0.               # N维向量：KKT条件 αi=0   ←→ yi*g(xi)≥1
            violateKKT_[(0<α_) & (α_<C) & (yg_==1)] = 0.       # N维向量：KKT条件 0<αi<C ←→ yi*g(xi)=1
            violateKKT_[(α_==C) & (yg_<=1)] = 0.               # N维向量：KKT条件 αi=C   ←→ yi*g(xi)≤1
            if violateKKT_.max()<self.tol:
                print(f'第{t}次SMO迭代，最大违反KKT条件的程度达到收敛精度{self.tol}，停止迭代!')
                break
            """选择αi"""
            indexViolateKKT_ = where(violateKKT_>0)[0]  # 数组索引：找出违反KKT条件的α
            indexNonBoundViolateKKT_ = intersect1d(indexViolateKKT_, indexNonBound_)  # 数组索引：找出违反KKT条件的非边界α
            if random()<0.85:
                # 有较大的概率（85%）选取违反KKT条件程度最大的αi进行下一步优化，若有非边界α，首选非边界α
                if len(indexNonBoundViolateKKT_)>0:
                    # 若存在违反KKT条件的非边界α，则选取违反KKT条件程度最大的非边界α
                    i = indexNonBoundViolateKKT_[violateKKT_[indexNonBoundViolateKKT_].argmax()]
                else:
                    # 若不存在违反KKT条件的非边界α，则直接选取违反KKT条件程度最大的α
                    i = violateKKT_.argmax()
            else:
                # 保留较小的概率（15%）随机选取违反KKT条件的α
                i = choice(indexViolateKKT_)
            """选择αj"""
            j = choice(indexViolateKKT_)  # 随机选择另一个违反KKT条件的αj
            # j = E_.argmin() if E_[i]>0 else E_.argmax()  # 经过试验发现：若按照“使|Ei-Ej|最大”这样的规则来选择j，则往往所需迭代次数极大，甚至陷入死循环。优化效率还不如随机选择j。
            while (X__[i]==X__[j]).all():
                j = choice(range(N))  # 所选样本X__[i]、X__[j]完全相同，重新选择αj
            """优化αi、αj"""
            # print(f'第{t}次SMO迭代，选择i = {i}, j = {j}')
            αiOld, αjOld = α_[i], α_[j]  # 记录αi、αj的旧值
            if y_[i]!=y_[j]:             # 确定αj的下限L、上限H
                L, H = max(0, αjOld - αiOld), min(C, C + αjOld - αiOld)
            else:
                L, H = max(0, αjOld + αiOld - C), min(C, αjOld + αiOld)
            Kii = K__[i, i]        # 从核矩阵读取核函数值
            Kjj = K__[j, j]        # 从核矩阵读取核函数值
            Kij = K__[i, j]        # 从核矩阵读取核函数值
            η = Kii + Kjj - 2*Kij  # ||φ(xi)-φ(xj)||**2 >= 0
            αj = αjOld + y_[j]*(E_[i] - E_[j])/η  # 未剪辑的αj
            if αj>H:    # 剪辑αj
                αj = H
            elif αj<L:
                αj = L
            else:
                pass
            αi = αiOld + y_[i]*y_[j]*(αjOld - αj)  # 未剪辑的αi
            if αi>C:    # 剪辑αi
                αi = C
            elif αi<0:
                αi = 0
            else:
                pass
            α_[j], α_[i] = αj, αi   # 更新αi、αj
            """更新目标函数值"""
            vi = g_[i] - αiOld*y_[i]*Kii - αjOld*y_[j]*Kij - b  # 记号vi
            vj = g_[j] - αiOld*y_[i]*Kij - αjOld*y_[j]*Kjj - b  # 记号vj
            minimizedObjectiveValue += (0.5*(αi**2 - αiOld**2)*Kii
                                      + 0.5*(αj**2 - αjOld**2)*Kjj
                                      + (αi*αj - αiOld*αjOld)*Kij*y_[i]*y_[j]
                                      + vi*y_[i]*(αi - αiOld)
                                      + vj*y_[j]*(αj - αjOld)
                                        ) - (αi - αiOld + αj - αjOld)  # 更新目标函数值
            """更新偏置b"""
            if 0<α_[i]<C:
                b = -E_[i] - y_[i]*Kii*(α_[i] - αiOld) - y_[j]*Kij*(α_[j] - αjOld) + b
            elif 0<α_[j]<C:
                b = -E_[j] - y_[i]*Kij*(α_[i] - αiOld) - y_[j]*Kjj*(α_[j] - αjOld) + b
            else:
                bi = -E_[i] - y_[i]*Kii*(α_[i] - αiOld) - y_[j]*Kij*(α_[j] - αjOld) + b
                bj = -E_[j] - y_[i]*Kij*(α_[i] - αiOld) - y_[j]*Kjj*(α_[j] - αjOld) + b
                b = (bi + bj)/2
            if isnan(b):
                raise ValueError('偏置b值为nan，排查错误！')
        else:
            print(f'达到最大迭代次数{self.maxIterations}!')

        """优化结束，计算偏置b"""
        indexSV_ = where(α_>0)[0]                   # 数组索引：支持向量
        self.αSV_ = α_[indexSV_]                    # 向量：支持向量对应的拉格朗日乘子α
        self.ySV_ = y_[indexSV_]                    # 向量：支持向量对应的标签
        self.supportVectors__ = X__[indexSV_]       # 矩阵：所有支持向量
        self.α_ = α_                                # N维向量：N个拉格朗日乘子
        indexNonBound_ = where((0<α_) & (α_<C))[0]  # 数组索引：满足0<α<C的支持向量
        if len(indexNonBound_)>0:
            # 若存在满足0<α<C的支持向量，计算偏置b，取平均值
            b_ = [(y_[k] - (self.αSV_*self.ySV_) @ K__[k, indexSV_]) for k in indexNonBound_]
            self.b = sum(b_)/len(b_)  # 取偏置b平均值
        else:
            self.b = b
            print('不存在满足0<α<C的α，取最后一次迭代得到的偏置b')
        """计算权重向量w_"""
        if self.kernel=='linear':
            # 若使用线性核函数，计算权重向量
            self.w_ = (self.αSV_*self.ySV_) @ self.supportVectors__

    def predict(self, X__: ndarray) -> ndarray:
        """测试"""
        assert type(X__)==ndarray and X__.ndim==2, '输入测试样本矩阵X__应为2维ndarray'
        assert X__.shape[1]==self.M, f'输入测试样本维数应等于训练样本维数{self.M}'
        y_ = where(self.decisionFunction(X__)>=0, 1, -1)  # 判定类别
        return y_

    def decisionFunction(self, X__: ndarray) -> ndarray:
        """计算决策函数值 f = w_ @ x_ + b"""
        assert X__.ndim==2, '输入样本矩阵X__应为2维ndarray'
        assert X__.shape[1]==self.M, f'输入测试样本维数应等于训练样本维数{self.M}'
        if self.solver=='smo':
            # 若使用SMO算法进行训练
            f_ = (self.αSV_*self.ySV_) @ self.kernelFunction(self.supportVectors__, X__)  + self.b
        return f_

    def accuracy(self, X__: ndarray, y_: ndarray) -> float:
        """计算测试正确率"""
        测试样本正确个数 = sum(self.predict(X__)==y_)
        测试样本总数 = len(y_)
        return 测试样本正确个数/测试样本总数
