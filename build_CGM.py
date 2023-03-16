from scipy.stats import multivariate_normal
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# path = 'foo.csv'
path = 'foo2.csv'
data = pd.read_csv(path, header=None)
X1 = data[0].values.reshape(-1, 1)
X2 = data[1].values.reshape(-1, 1)
X3 = data[2].values.reshape(-1, 1)
X = np.hstack((X1, X2, X3))  # 读取数据

plt.xlabel('X(meter)')
plt.ylabel('Y(meter)')
plt.title('Original Data')
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.show()

plt.xlabel('X(meter)')
plt.ylabel('Y(meter)')
plt.title('Channel Gain Map (Classic Model)')
im = plt.scatter(X[:, 0], X[:, 1], s=10, c=X[:, 2], cmap='jet_r')
cb = plt.colorbar(im)
cb.set_label('Path Loss(dB)')
plt.show()  # 绘制原始数据的散点图


def lg_distance(X):  # 欧几里得范数项处理
    A = np.square(X)
    Q = np.sqrt(A[:, 0] + A[:, 1])
    D = 10 * np.log10(Q)
    return D


def update_W(X, Beta, Alpha, Var, Pi):  # 更新隐变量分布
    n_points, n_clusters = len(X), len(Pi)
    pdfs = np.zeros((n_points, n_clusters))
    for i in range(n_clusters):
        for j in range(n_points):
            pdfs[j, i] = Pi[i] * multivariate_normal.pdf(X[j, 2], Beta[i] + Alpha[i] * D[j], Var[i])  # 见论文（2）式
    W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
    return W


def update_Pi(W):  # 更新每一个信道模型的比重
    Pi = W.sum(axis=0) / W.sum()
    return Pi


def update_Alpha(W, D, X):  # 更新参数α，见定理3.1
    n_clusters = W.shape[1]
    Alpha = np.zeros(n_clusters)
    for i in range(n_clusters):
        dkrk = np.sum(W[:, i] * D * X[:, 2]) / np.sum(W[:, i])
        dk = np.sum(W[:, i] * D) / np.sum(W[:, i])
        dk2 = np.sum(W[:, i] * np.square(D)) / np.sum(W[:, i])
        rk = np.sum(W[:, i] * X[:, 2]) / np.sum(W[:, i])
        Alpha[i] = (dkrk - dk * rk) / (dk2 - dk ** 2)
    return Alpha


def update_Beta(W, D, X):  # 更新参数β，见定理3.1
    n_clusters = W.shape[1]
    Beta = np.zeros(n_clusters)
    for i in range(n_clusters):
        dkrk = np.sum(W[:, i] * D * X[:, 2]) / np.sum(W[:, i])
        dk = np.sum(W[:, i] * D) / np.sum(W[:, i])
        dk2 = np.sum(W[:, i] * np.square(D)) / np.sum(W[:, i])
        rk = np.sum(W[:, i] * X[:, 2]) / np.sum(W[:, i])
        Beta[i] = (dk2 * rk - dk * dkrk) / (dk2 - dk ** 2)
    return Beta


def update_Var(W, D, X, Alpha, Beta):  # 更新s内的方差，见定理3.1
    n_clusters = W.shape[1]
    Var = np.zeros(n_clusters)
    for i in range(n_clusters):
        Var[i] = np.sum(W[:, i] * np.square(X[:, 2] - Beta[i] - Alpha[i] * D)) / np.sum(W[:, i])
    return Var


def logLH(X, Pi, Alpha, Beta, Var):  # 定义极大似然函数，用来停止迭代
    n_points, n_clusters = len(X), len(Pi)
    pdfs = np.zeros((n_points, n_clusters))
    for i in range(n_clusters):
        for j in range(n_points):
            pdfs[j, i] = Pi[i] * multivariate_normal.pdf(X[j, 2], Beta[i] + Alpha[i] * D[j], Var[i])
    return np.mean(np.log(pdfs.sum(axis=1)))


if __name__ == '__main__':
    n_clusters = 5
    n_points = len(X)

    Alpha = [2, 3, 3.5, 4, 4.5]
    Beta = [30, 50, 85, 110, 130]
    Var = [6, 13, 15, 9.5, 7]  # 初始参数
    W = np.ones((n_points, n_clusters)) / n_clusters  # 2000行[0.2 0.2 0.2 0.2 0.2]
    Pi = W.sum(axis=0) / W.sum()  # [0.2, 0.2, 0.2, 0.2, 0.2]

    iteration = 0
    iterations = 100
    tol = 1e-4
    loglh = [0]

    while iteration < iterations:  # 迭代次数限制
        D = lg_distance(X)
        W = update_W(X, Beta, Alpha, Var, Pi)
        Alpha = update_Alpha(W, D, X)
        Beta = update_Beta(W, D, X)
        Var = update_Var(W, D, X, Alpha, Beta)
        Pi = update_Pi(W)

        loglh.append(logLH(X, Pi, Alpha, Beta, Var))
        change = np.abs(loglh[-1] - loglh[-2])

        print('log-likelihood:%.3f' % loglh[-1])

        if change < tol:  # 停止迭代
            max_index_axis1 = np.argmax(W, axis=1)  # 得到每个点受哪个模型影响的索引
            Y = np.zeros(n_points)

            gain_dif = 0

            for i in range(n_points):  # 用em算法训练后的参数得到输出值
                #if -400 <= X[i, 0] <= 400 and -300 <= X[i, 1] <= -100:
                    #mix_ratio = (X[i, 1] + 300) / 200
                    #Y[i] = mix_ratio * (Beta[0] + D[i] * Alpha[0] + np.random.normal(0, np.sqrt(Var[0]))) + (1 - mix_ratio) * (Beta[4] + D[i] * Alpha[4] + np.random.normal(0, np.sqrt(Var[4])))
                #else:
                Y[i] = Beta[max_index_axis1[i]] + D[i] * Alpha[max_index_axis1[i]] + np.random.normal(0, np.sqrt(Var[max_index_axis1[i]]))
                gain_dif = gain_dif + (Y[i] - X[i, 2]) ** 2

            RMSE = (gain_dif / n_points) ** 0.5
            NRMSE = RMSE / (max(Y) - min(Y))
            print('NRMSE:%.3f' % NRMSE)

            plt.xlabel('X(meter)')
            plt.ylabel('Y(meter)')
            plt.title('Path Loss of Trained Data')
            im = plt.scatter(X[:, 0], X[:, 1], s=10, c=Y, cmap='jet_r')
            cb = plt.colorbar(im)
            cb.set_label('Path Loss(dB)')
            plt.show()  # 绘制训练后的散点图

            points = []
            for i in range(len(X)):
                point = [X[i, 0], X[i, 1]]
                points.append(point)
            points = np.array(points)

            xi = np.linspace(min(X[:, 0]), max(X[:, 0]), 2000)
            yi = np.linspace(min(X[:, 1]), max(X[:, 1]), 2000)  # 创造等差数列
            xi, yi = np.meshgrid(xi, yi)  # 生成网格点坐标矩阵
            zi = griddata(points, Y, (xi, yi), method='nearest')  # 给训练后的离散数据进行插值
            # zi = griddata(points, Y, (xi, yi), method='linear')  # 给训练后的离散数据进行插值

            plt.xlabel('X(meter)')
            plt.ylabel('Y(meter)')
            plt.title('Channel Gain Map')
            im = plt.contourf(xi, yi, zi, 100, cmap='jet_r')  # 绘制等高线图(填充轮廓)
            cb = plt.colorbar(im)
            cb.set_label('Path Loss(dB)')
            plt.show()
            break

        iteration = iteration + 1
