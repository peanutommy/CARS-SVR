import copy
import numpy as np
from hyperopt import fmin, tpe, hp
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import kennard_stone_algorithm
from cars_visualize import add_visualizations_to_cars


def CARS_Cloud(X, y,spectrum_range, N=40, cv=5):
    '''
    优化后的CARS算法实现
    '''
    # 数据预处理
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = y.ravel()

    p = 0.8  # 训练集比例
    m, n = X.shape

    # 修改采样参数以确保更快的特征数量减少
    u = np.power((n / 10), (1 / (N - 1)))
    k = (1 / (N - 1)) * np.log(n / 10)
    cal_num = int(np.round(m * p))

    b2 = np.array(spectrum_range)  # 使用实际的光谱范围值
    x = copy.deepcopy(X)
    D = np.vstack((np.array(b2).reshape(1, -1), X))

    # 预分配数组
    WaveNum = []
    RMSECV = []
    selected_features = []
    current_selected = np.arange(n)  # 跟踪当前选择的特征
    # 随机采样
    for i in range(N):
        # 计算当前迭代的波长数
        r = u * np.exp(-k * (i + 1))
        wave_num = max(5, int(np.round(r * n)))  # 确保至少保留5个变量
        WaveNum.append(wave_num)
        cal_index = np.random.choice(np.arange(m), size=cal_num, replace=False)
        # svr建模
        def svr_params(x, y, cv):
            '''
            交叉验证函数
            '''
            RMSE = []
            space = {'c': hp.uniform('c', 450, 480)}

            def objective(params):
                kf = KFold(n_splits=cv, shuffle=True, random_state=42)
                rmse_list = []
                c = params['c']
                for train_index, test_index in kf.split(x):
                    x_train, x_test = x[train_index], x[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    svr = SVR(kernel='linear', C=c)
                    svr.fit(x_train, y_train)
                    y_predict = svr.predict(x_test)
                    rmse_list.append(mean_squared_error(y_test, y_predict))
                return np.mean(rmse_list)

            best = fmin(objective, space, tpe.suggest, max_evals=10)
            print(best)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
            best_c = best['c']
            svr = SVR(kernel='linear', C=best_c)
            svr.fit(x_train, y_train)
            y_predict = svr.predict(x_test)
            RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
            return RMSE, best

        # 交叉验证
        rmse, best = svr_params(x[cal_index], y[cal_index], cv)
        RMSECV.append(np.mean(rmse))  # 使用平均RMSE

        c = best['c']
        svr = SVR(kernel='linear', C=c)
        svr.fit(x[cal_index], y[cal_index])
        weights = svr.coef_[0]

        # 获取变量重要性
        coef = np.abs(weights).ravel()
        importance_order = np.argsort(-coef)
        selected_vars = importance_order[:wave_num]

        # 保存当前选择的特征
        current_selected = b2[selected_vars]
        selected_features.append(current_selected.copy())

        # 更新数据
        x = x[:, selected_vars]
        b2 = b2[selected_vars]

    # 找到最优解
    RMSECV = np.array(RMSECV)
    valid_indices = ~np.isinf(RMSECV)

    # 修复：使用 any() 来检查是否有有效的解
    if not valid_indices.any():
        raise ValueError("No valid solution found")

    MinIndex = np.argmin(RMSECV)
    best_features = selected_features[MinIndex]

    # 绘图
    add_visualizations_to_cars(WaveNum, RMSECV, selected_features, MinIndex, svr)

    return best_features, MinIndex + 1, RMSECV[MinIndex], WaveNum[MinIndex]


if __name__ == "__main__":
    import pandas as pd

    # 读取数据
    df = pd.read_excel('sum_amylopectin_spectrum.xlsx', header=None)
    X = df.iloc[1:, :-1].values
    y = df.iloc[1:, -1].values
    spectrum_range = df.iloc[0, :-1].values
    y = np.array(y)
    print('shape of x:', X.shape)
    print('shape of y:', y.shape)

    # 运行CARS算法
    optimal_features, best_iteration, best_rmsecv, n_features = CARS_Cloud(X, y, spectrum_range)
    wavelength_to_column = {wavelength: idx for idx, wavelength in enumerate(spectrum_range)}
    column_indices = [wavelength_to_column[wavelength] for wavelength in optimal_features]
    x_selected = X[:, column_indices]
    train_index, test_index = kennard_stone_algorithm.kennardstonealgorithm(X, 100)
    x_train, x_test = x_selected[train_index], x_selected[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # x_train, x_test, y_train, y_test = train_test_split(x_selected, y, test_size=0.3, random_state=1)

    space = {'c': hp.randint('c', 1, 30)}


    def plsr(params):
        c = params['c']
        plsr = PLSRegression(n_components=c)
        plsr.fit(x_train, y_train)
        y_pred = plsr.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse


    # 使用TPE算法进行超参数优化，最大评估次数为100
    best = fmin(plsr, space, tpe.suggest, max_evals=200)
    print('\nBest Hyperparameters:', best)
    c = int(best['c'])
    plsr = PLSRegression(n_components=c)
    plsr.fit(x_train, y_train)
    y_train_pred = plsr.predict(x_train)
    y_pred = plsr.predict(x_test)

    print("\n结果汇总:")
    print(f"最佳迭代次数: {best_iteration}")
    print(f"最佳RMSECV值: {best_rmsecv:.4f}")
    print(f"选中的特征数量: {n_features}")
    print(f"选中特征的重要性比例: {(n_features / X.shape[1] * 100):.2f}%")
    print("\n选中的特征索引:", optimal_features)
    # 评估模型
    rb = r2_score(y_train, y_train_pred)
    rs = r2_score(y_test, y_pred)
    rs_mse = mean_squared_error(y_test, y_pred)
    rs_mae = mean_absolute_error(y_test, y_pred)
    rb_mse = mean_squared_error(y_train, y_train_pred)
    rb_mae = mean_absolute_error(y_train, y_train_pred)

    print("train均方误差(mse):", rb_mse)
    print("train平均绝对误差(mae):", rs_mae)
    print("trainR2", rb)
    print("test均方误差(mse):", rs_mse)
    print("test平均绝对误差(mae):", rs_mae)
    print("testR方(r2):", rs)
"""
    # 可视化
    import seaborn as sns

    # 创建一个包含训练集和测试集真实值与预测值的数据框
    data_train = pd.DataFrame({'True': y_train, 'Predicted': y_train_pred, 'Data Set': 'Train'})
    data_test = pd.DataFrame({'True': y_test, 'Predicted': y_pred, 'Data Set': 'Test'})
    data = pd.concat([data_train, data_test])
    # 自定义调色板
    palette = {'Train': '#b4d4e1', 'Test': '#f4ba8a'}
    # 创建 JointGrid 对象
    plt.figure(figsize=(8, 6), dpi=1200)
    g = sns.JointGrid(data=data, x="True", y="Predicted", hue="Data Set", height=10, palette=palette)
    # 绘制中心的散点图
    g.plot_joint(sns.scatterplot, alpha=1, s=50)
    # 添加训练集的回归线
    sns.regplot(data=data_train, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color='#b4d4e1',
                label='Train Regression Line')
    # 添加测试集的回归线
    sns.regplot(data=data_test, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color='#f4ba8a',
                label='Test Regression Line')
    # 添加边缘的柱状图
    g.plot_marginals(sns.histplot, kde=False, element='bars', multiple='stack', alpha=0.5)
    # 添加拟合优度文本在右下角
    ax = g.ax_joint
    ax.text(0.95, 0.1, f'Train $R^2$ = {rb:.3f}', transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
            horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    ax.text(0.95, 0.05, f'Test $R^2$ = {rs:.3f}', transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
            horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    # 在左上角添加模型名称文本
    ax.text(0.65, 0.99, 'Model = CARS-SVR', transform=ax.transAxes, fontsize=12, verticalalignment='top',
            horizontalalignment='left', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    # 添加中心线
    ax.plot([data['True'].min(), data['True'].max()], [data['True'].min(), data['True'].max()], c="black", alpha=0.5,
            linestyle='--', label='x=y')
    ax.legend()
    plt.savefig("TrueFalse.png", format='png', bbox_inches='tight')
    plt.show()
    """