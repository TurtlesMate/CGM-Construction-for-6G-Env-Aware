import matplotlib.pyplot as plt
import numpy as np

all_points = np.random.rand(2000, 2)  # 第一个参数表示随机生成数据点的数目，第二个参数表示数据点是一个具有二维特征的
test_points = np.zeros((2000, 2))
test_points[:, 0] = (all_points[:, 0] - 0.5) * 2000
test_points[:, 1] = (all_points[:, 1] - 0.5) * 2000

z = np.zeros(2000)
A = np.square(test_points)
Q = np.sqrt(A[:, 0] + A[:, 1])
D = 10 * np.log10(Q)

for i in range(2000):
    dist1 = (2 / 3 * test_points[i, 0] + test_points[i, 1]) / np.sqrt(13 / 9)
    dist2 = (-0.5 * test_points[i, 0] + test_points[i, 1]) / np.sqrt(1.25)
    dist3 = (0.5 * test_points[i, 0] + test_points[i, 1]) / np.sqrt(1.25)

    if -400 <= test_points[i, 0] <= 400 and -300 <= test_points[i, 1] <= -100:
        mix_ratio = (test_points[i, 1] + 300) / 200
        z[i] = mix_ratio * (30 + 2.2 * D[i] + np.random.normal(0, np.sqrt(6.25))) + (1 - mix_ratio) * (
                130 + 4.1 * D[i] + np.random.normal(0, np.sqrt(7.84)))

    elif -400 <= test_points[i, 0] <= 400 and -700 <= test_points[i, 1] <= -500:
        mix_ratio = (test_points[i, 1] + 700) / 200
        z[i] = mix_ratio * (130 + 4.1 * D[i] + np.random.normal(0, np.sqrt(7.84))) + (1 - mix_ratio) * (
                80 + 3.1 * D[i] + np.random.normal(0, np.sqrt(10.24)))

    elif -500 <= test_points[i, 0] <= -300 and -600 <= test_points[i, 1] <= -200:
        mix_ratio = (test_points[i, 0] + 500) / 200
        z[i] = mix_ratio * (130 + 4.1 * D[i] + np.random.normal(0, np.sqrt(7.84))) + (1 - mix_ratio) * (
                80 + 3.1 * D[i] + np.random.normal(0, np.sqrt(10.24)))

    elif 300 <= test_points[i, 0] <= 500 and -600 <= test_points[i, 1] <= -200:
        mix_ratio = (test_points[i, 0] - 300) / 200
        z[i] = mix_ratio * (80 + 3.1 * D[i] + np.random.normal(0, np.sqrt(10.24))) + (1 - mix_ratio) * (
                130 + 4.1 * D[i] + np.random.normal(0, np.sqrt(7.84)))

    elif -600 <= test_points[i, 0] <= 0 and 300 <= test_points[i, 1] <= 500:
        mix_ratio = (test_points[i, 1] - 300) / 200
        z[i] = mix_ratio * (105 + 3.6 * D[i] + np.random.normal(0, np.sqrt(7.84))) + (1 - mix_ratio) * (
                30 + 2.2 * D[i] + np.random.normal(0, np.sqrt(6.25)))

    elif -600 <= test_points[i, 0] <= 0 and 750 <= test_points[i, 1] <= 850:
        mix_ratio = (test_points[i, 1] - 750) / 100
        z[i] = mix_ratio * (55 + 2.6 * D[i] + np.random.normal(0, np.sqrt(10.24))) + (1 - mix_ratio) * (
                105 + 3.6 * D[i] + np.random.normal(0, np.sqrt(7.84)))

    elif -650 <= test_points[i, 0] <= -550 and 400 <= test_points[i, 1] <= 800:
        mix_ratio = (test_points[i, 0] + 650) / 100
        z[i] = mix_ratio * (105 + 3.6 * D[i] + np.random.normal(0, np.sqrt(7.84))) + (1 - mix_ratio) * (
                55 + 2.6 * D[i] + np.random.normal(0, np.sqrt(10.24)))

    elif -100 <= test_points[i, 0] <= 100 and 400 <= test_points[i, 1] <= 800:
        mix_ratio = (test_points[i, 0] + 100) / 200
        z[i] = mix_ratio * (30 + 2.2 * D[i] + np.random.normal(0, np.sqrt(6.25))) + (1 - mix_ratio) * (
                130 + 4.1 * D[i] + np.random.normal(0, np.sqrt(7.84)))

    elif -50 <= test_points[i, 0] <= 50 and 800 <= test_points[i, 1] <= 1000:
        mix_ratio = (test_points[i, 0] + 50) / 100
        z[i] = mix_ratio * (30 + 2.2 * D[i] + np.random.normal(0, np.sqrt(6.25))) + (1 - mix_ratio) * (
                55 + 2.6 * D[i] + np.random.normal(0, np.sqrt(10.24)))

    elif test_points[i, 0] <= -600 and -50 <= dist1 <= 50:
        mix_ratio = (dist1 + 50) / 100
        z[i] = mix_ratio * (55 + 2.6 * D[i] + np.random.normal(0, np.sqrt(10.24))) + (1 - mix_ratio) * (
                30 + 2.2 * D[i] + np.random.normal(0, np.sqrt(6.25)))

    elif test_points[i, 0] <= -400 and -100 <= dist2 <= 100:
        mix_ratio = (dist2 + 100) / 200
        z[i] = mix_ratio * (30 + 2.2 * D[i] + np.random.normal(0, np.sqrt(6.25))) + (1 - mix_ratio) * (
                80 + 3.1 * D[i] + np.random.normal(0, np.sqrt(10.24)))

    elif test_points[i, 0] >= 400 and -100 <= dist3 <= 100:
        mix_ratio = (dist3 + 100) / 200
        z[i] = mix_ratio * (30 + 2.2 * D[i] + np.random.normal(0, np.sqrt(6.25))) + (1 - mix_ratio) * (
                80 + 3.1 * D[i] + np.random.normal(0, np.sqrt(10.24)))

    elif -600 <= test_points[i, 0] <= 0 and 400 <= test_points[i, 1] <= 800:  # Indoor 1
        z[i] = 105 + 3.6 * D[i] + np.random.normal(0, np.sqrt(7.84))

    elif -400 <= test_points[i, 0] <= 400 and -600 <= test_points[i, 1] <= -200:  # Indoor2
        z[i] = 130 + 4.1 * D[i] + np.random.normal(0, np.sqrt(7.84))

    elif (-750 <= test_points[i, 0] <= -600 and test_points[i, 1] >= 400) or (
            test_points[i, 0] < -750 and test_points[i, 1] >= -0.8 * test_points[i, 0] - 200) or (
            -600 <= test_points[i, 0] <= 0 and test_points[i, 1] > 800):  # NLos1
        z[i] = 55 + 2.6 * D[i] + np.random.normal(0, np.sqrt(10.24))

    elif (test_points[i, 0] < -400 and test_points[i, 1] <= 0.67 * test_points[i, 0] + 66) or (
            test_points[i, 0] > 400 and test_points[i, 1] <= -0.67 * test_points[i, 0] + 66) or (
            -400 <= test_points[i, 0] <= 400 and test_points[i, 1] < -600):  # NLos2
        z[i] = 80 + 3.1 * D[i] + np.random.normal(0, np.sqrt(10.24))

    else:  # Los
        z[i] = 30 + 2.2 * D[i] + np.random.normal(0, np.sqrt(6.25))

# print(z)
# pl.plot(test_points[:, 0], test_points[:, 1],  'b.')  # 绘制出这些数据点，以点的形式

X1 = test_points[:, 0].reshape(-1, 1)
X2 = test_points[:, 1].reshape(-1, 1)
X3 = z.reshape(-1, 1)
X = np.hstack((X1, X2, X3))

# print(X)

plt.xlabel('X(meter)')
plt.ylabel('y(meter)')
im = plt.scatter(X[:, 0], X[:, 1], s=10, c=X[:, 2], cmap='jet_r')
cb = plt.colorbar(im)
cb.set_label('colorbar')
plt.show()

# np.savetxt("foo2.csv", X, delimiter=",")
