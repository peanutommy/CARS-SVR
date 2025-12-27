import copy
import numpy as np
from hyperopt import fmin, tpe, hp
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def CARS_Cloud(X, y,spectrum_range, N=40, cv=5):
    
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = y.ravel()

    p = 0.8
    m, n = X.shape

    u = np.power((n / 10), (1 / (N - 1)))
    k = (1 / (N - 1)) * np.log(n / 10)
    cal_num = int(np.round(m * p))

    b2 = np.array(spectrum_range)
    x = copy.deepcopy(X)
    D = np.vstack((np.array(b2).reshape(1, -1), X))
    WaveNum = []
    RMSECV = []
    selected_features = []
    current_selected = np.arange(n)
    for i in range(N):
        r = u * np.exp(-k * (i + 1))
        wave_num = max(5, int(np.round(r * n)))
        WaveNum.append(wave_num)
        cal_index = np.random.choice(np.arange(m), size=cal_num, replace=False)
        def svr_params(x, y, cv):
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

        rmse, best = svr_params(x[cal_index], y[cal_index], cv)
        RMSECV.append(np.mean(rmse)) 

        c = best['c']
        svr = SVR(kernel='linear', C=c)
        svr.fit(x[cal_index], y[cal_index])
        weights = svr.coef_[0]

        coef = np.abs(weights).ravel()
        importance_order = np.argsort(-coef)
        selected_vars = importance_order[:wave_num]

        current_selected = b2[selected_vars]
        selected_features.append(current_selected.copy())

        x = x[:, selected_vars]
        b2 = b2[selected_vars]


    RMSECV = np.array(RMSECV)
    valid_indices = ~np.isinf(RMSECV)

    if not valid_indices.any():
        raise ValueError("No valid solution found")

    MinIndex = np.argmin(RMSECV)
    best_features = selected_features[MinIndex]


    return best_features, MinIndex + 1, RMSECV[MinIndex], WaveNum[MinIndex]

    """
