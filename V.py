import xlrd2
from gurobipy import *
import gurobipy as gp
import numpy as np
import pandas as pd

cycle_time = 30
C0 = 150
ratio = 3
Num_of_Rolling_days = 21
Num_of_SAA_samples = 35
Num_of_periods = 24
Num_of_decisions = int(Num_of_periods * 30 / cycle_time)
beta = 0.05
Num_of_types = Num_of_periods

P = 'FlatPrice'
# P = 'VaryingPrice'
Price = {}
if P == 'FlatPrice':
    h = 2
    for t in range(0, Num_of_decisions):
       for k in range(0, Num_of_types):
           Price[t, k] = (k + 1)
elif P == 'VaryingPrice':
    h = 5
    for tao in range(0, Num_of_periods):
        if tao in range(4, 8) or tao in range(18, 22):
            for k in range(0, Num_of_types):
                Price[tao, k] = (k + 1) * 3
        else:
            for k in range(0, Num_of_types):
                Price[tao, k] = k + 1

P0 = 0.65
P1 = 0.25
P2 = 0.1
PAI = pd.DataFrame({
    0: np.array([0.7, 0.2, 0.1], dtype='float32'),
    1: np.array([0.7, 0.2, 0.1], dtype='float32'),
    2: np.array([0.7, 0.2, 0.1], dtype='float32'),
    3: np.array([0.7, 0.2, 0.1], dtype='float32'),
    4: np.array([0.7, 0.2, 0.1], dtype='float32'),
    5: np.array([0.7, 0.2, 0.1], dtype='float32'),
    6: np.array([0.7, 0.2, 0.1], dtype='float32'),
    7: np.array([0.7, 0.2, 0.1], dtype='float32'),
    8: np.array([0.7, 0.2, 0.1], dtype='float32'),
    9: np.array([0.7, 0.2, 0.1], dtype='float32'),
    10: np.array([0.7, 0.2, 0.1], dtype='float32'),
    11: np.array([0.7, 0.2, 0.1], dtype='float32'),
    12: np.array([0.7, 0.2, 0.1], dtype='float32'),
    13: np.array([0.7, 0.2, 0.1], dtype='float32'),
    14: np.array([0.7, 0.2, 0.1], dtype='float32'),
    15: np.array([0.7, 0.2, 0.1], dtype='float32'),
    16: np.array([0.7, 0.2, 0.1], dtype='float32'),
    17: np.array([0.7, 0.2, 0.1], dtype='float32'),
    18: np.array([0.7, 0.2, 0.1], dtype='float32'),
    19: np.array([0.7, 0.2, 0.1], dtype='float32'),
    20: np.array([0.7, 0.2, 0.1], dtype='float32'),
    21: np.array([0.7, 0.2, 0.1], dtype='float32'),
    22: np.array([0.7, 0.2, 0.1], dtype='float32'),
    23: np.array([0.7, 0.2, 0.1], dtype='float32'),
}, index=[0, 1, 2])


def SortDK(tao, dict):
    l1 = dict.index
    r = pd.DataFrame()
    if tao <= Num_of_periods - 2:
        l2 = dict.columns
        for column in l2:
            list = {}
            for index in l1:
                r.loc[column, index] = index
                list[index] = dict.loc[index, column]
            l = len(list)
            for i in range(0, l - 1):
                for j in range(i, l):
                    if list[i] > list[j]:
                        r.loc[column, i], r.loc[column, j] = r.loc[column, j], r.loc[column, i]
                        list[i], list[j] = list[j], list[i]
        return r[range(0, int(Num_of_SAA_samples * beta) + 1)]


def Get_Xtk(tao, Ctao, Dtao, Gxdelay):
    BigM = 10000
    N = Num_of_SAA_samples
    T = Num_of_periods
    K = Num_of_types
    Xtypes = tuplelist(range(0, min(K, T - tao)))
    Yperiods = tuplelist(range(tao + 1, tao + 1 + min(h, T - tao - 1)))
    Ytypes = tuplelist(range(0, min(K, T - tao - 1)))
    Samples = tuplelist(range(0, N))

    datafirst24 = xlrd2.open_workbook('Files/Data/24New' + str(ratio) + 'records' + str(Num_of_SAA_samples) + '.xls')
    Demand_Sample = datafirst24.sheet_by_name("Newdata")

    # 建立模型
    model = gp.Model("Get_XandY")

    # X变量
    X = model.addVars(Xtypes, vtype=GRB.INTEGER, name='X')
    Mx = model.addVars(Xtypes, vtype=GRB.INTEGER, name='Mx')
    B1 = model.addVars(Xtypes, vtype=GRB.BINARY, name='B1')
    B2 = model.addVars(Xtypes, vtype=GRB.BINARY, name='B2')

    # Xtk约束
    model.addConstr(X.sum() <= Ctao, name='b')
    for k in Xtypes:
        model.addConstr(X[k] <= Dtao[k], name='c')

    # tao时刻匹配
    for k in Xtypes:
        model.addConstr(Mx[k] <= X[k], name=str(k))
        model.addConstr(Mx[k] <= Dtao[k], name=str(k))
        model.addConstr(Mx[k] >= X[k] - BigM * B1[k], name=str(k))
        model.addConstr(Mx[k] >= Dtao[k] - BigM * B2[k], name=str(k))
        model.addConstr(B1[k] + B2[k] == 1, name=str(k))

    # Y变量
    Y = model.addVars(Yperiods, Ytypes, vtype=GRB.INTEGER, name='Y')
    My = model.addVars(Samples, Yperiods, Ytypes, vtype=GRB.INTEGER, name='My')
    B3 = model.addVars(Samples, Yperiods, Ytypes, vtype=GRB.BINARY, name='B3')
    B4 = model.addVars(Samples, Yperiods, Ytypes, vtype=GRB.BINARY, name='B4')
    y = model.addVars(Samples, Yperiods, vtype=GRB.BINARY, name='y')
    Z = model.addVars(Samples, vtype=GRB.BINARY, name='Z')

    # N次抽样需求
    demand = tupledict()
    Dnt = pd.DataFrame()
    for n in Samples:
        for t in Yperiods:
            for k in range(0, min(K, T - t)):
                demand[n, t, k] = round(Demand_Sample.cell_value(T * n + t, k))
            Dnt.loc[n, t] = sum(demand[n, t, k] for k in range(0, min(K, T - t)))
    miu = SortDK(tao, Dnt)  # 根据依据排序

    # 车位状态转移
    M = {}
    C = {}
    Gy = {}
    Gydelay = {}
    for n in Samples:
        for t in Yperiods:
            # 返回车位数量
            if t == tao + 1:
                Gydelay[n, t, 0] = Mx[0] * PAI.loc[0, 0]
                Gydelay[n, t, 1] = 0
                Gydelay[n, t, 2] = 0
            elif t == tao + 2:
                Gydelay[n, t, 0] = Mx[1] * PAI.loc[0, 1] + gp.quicksum(
                    My[n, e, t - e - 1] * PAI.loc[0, t - e - 1] for e in range(tao + 1, t))
                Gydelay[n, t, 1] = Mx[0] * PAI.loc[1, 0]
                Gydelay[n, t, 2] = 0
            elif t == tao + 3:
                Gydelay[n, t, 0] = Mx[2] * PAI.loc[0, 2] + gp.quicksum(
                    My[n, e, t - e - 1] * PAI.loc[0, t - e - 1] for e in range(tao + 1, t))
                Gydelay[n, t, 1] = Mx[1] * PAI.loc[1, 1] + gp.quicksum(
                    My[n, e, t - e - 2] * PAI.loc[1, t - e - 2] for e in range(tao + 1, t - 1))
                Gydelay[n, t, 2] = Mx[0] * PAI.loc[2, 0]
            else:
                Gydelay[n, t, 0] = Mx[t - tao - 1] * PAI.loc[0, t - tao - 1] + gp.quicksum(
                    My[n, e, t - e - 1] * PAI.loc[0, t - e - 1] for e in range(tao + 1, t))
                Gydelay[n, t, 1] = Mx[t - tao - 2] * PAI.loc[1, t - tao - 2] + gp.quicksum(
                    My[n, e, t - e - 2] * PAI.loc[1, t - e - 2] for e in range(tao + 1, t - 1))
                Gydelay[n, t, 2] = Mx[t - tao - 3] * PAI.loc[2, t - tao - 3] + gp.quicksum(
                    My[n, e, t - e - 3] * PAI.loc[2, t - e - 3] for e in range(tao + 1, t - 2))
            Gy[n, t] = Gydelay[n, t, 0] + Gydelay[n, t, 1] + Gydelay[n, t, 2] + Gxdelay[t]

            # Y匹配数量
            for k in Ytypes:
                # match约束
                if k >= T - t:
                    model.addConstr(Y[t, k] == 0, name=str(n) + str(t) + str(k))
                    model.addConstr(My[n, t, k] == 0, name=str(n) + str(t) + str(k))
                elif demand[n, t, k] == 0:
                    model.addConstr(My[n, t, k] == 0, name=str(n) + str(t) + str(k))
                else:
                    model.addConstr(My[n, t, k] <= Y[t, k], name=str(n) + str(t) + str(k))
                    model.addConstr(My[n, t, k] <= demand[n, t, k], name=str(n) + str(t) + str(k))
                    model.addConstr(My[n, t, k] >= Y[t, k] - BigM * B3[n, t, k], name=str(n) + str(t) + str(k))
                    model.addConstr(My[n, t, k] >= demand[n, t, k] - BigM * B4[n, t, k], name=str(n) + str(t) + str(k))
                    model.addConstr(B3[n, t, k] + B4[n, t, k] == 1, name=str(n) + str(t) + str(k))

            # 匹配数量以及库存数量
            M[n, t] = gp.quicksum(My[n, s, k] for s in range(tao + 1, t) for k in range(0, T - s))
            C[n, t] = Ctao - gp.quicksum(Mx[k] for k in Xtypes) - M[n, t] + gp.quicksum(
                Gy[n, s] for s in range(tao + 1, t + 1))

            # 库存约束
            model.addConstr(Y.sum(t, tuplelist(range(0, min(K, T - t)))) <= C[n, t], name="c")

    # d, e, f
    for t in Yperiods:
        sortedmiu = tuplelist(miu.loc[t, 0:int(N * beta) - 1].sort_values(ascending=False))
        remainedSamples = tuplelist(range(0, N))
        for j in range(0, int(N * beta)):
            remainedSamples.pop(int(sortedmiu[j]))
        model.addConstrs(y[i, t] == 0 for i in remainedSamples)

    for t in Yperiods:
        Sigma = gp.quicksum((Dnt.loc[miu.loc[t, j], t] - Dnt.loc[miu.loc[t, j + 1], t]) * y[miu.loc[t, j], t] for j in
                            range(0, int(N * beta)))
        model.addConstr(Y.sum(t, Ytypes) + Sigma <= Dnt.loc[miu.loc[t, 0], t], name="d")
        for j in range(0, int(N * beta)):
            model.addConstr(y[miu.loc[t, j], t] >= y[miu.loc[t, j + 1], t], name="e")
            model.addConstr(Z[miu.loc[t, j]] >= y[miu.loc[t, j], t], name="f")

    # 机会约束
    model.addConstr(gp.quicksum(Z[n] for n in Samples) <= int(N * beta), name="g")

    model.update()

    # 目标函数
    rt = gp.quicksum(Mx[k] * Price[tao, k] for k in Xtypes)
    ERt = gp.quicksum(My[n, t, k] * Price[t, k] for n in Samples for t in Yperiods for k in Ytypes) / N
    ERGt = gp.quicksum(Gydelay[n, t, j] * Q[t, j] for n in Samples for t in Yperiods for j in range(0, 3)) / N
    model.setObjective(rt + ERt + ERGt, GRB.MAXIMIZE)

    # 模型求解参数
    model.Params.MIPFocus = 3
    model.Params.Method = 1
    model.write("Files/LP/" + str(tao) + ".lp")

    model.optimize()

    # 打印结果
    if model.status == GRB.OPTIMAL:
        solution1 = model.getAttr('X', X)
        Z = model.getAttr('X', Z)
        y = model.getAttr('X', y)
        for i in Samples:
            if Z[i] == 1:
                print('Z', i)
            for j in Yperiods:
                if y[i, j] == 1:
                    print('y:', j)
        for j in Yperiods:
            for k in Ytypes:
                print(Y[j, k])
        print(model.objVal)
        runtime = model.Runtime
        return solution1, runtime, model.objVal


def Rolling(minute):
    # 实验集需求
    datasecond = xlrd2.open_workbook(
        'Files/Data/T' + str(int(24 * 30 / minute)) + 'New' + str(ratio) + 'records' + str(
            Num_of_Rolling_days) + '.xls')
    Demand_Test = datasecond.sheet_by_name("Newdata")
    demand = pd.DataFrame(np.zeros(Num_of_decisions * Num_of_types).reshape((Num_of_decisions, Num_of_types)),
                          dtype=int)

    R = {}
    FO = {}
    US = {}
    for n in range(0, Num_of_Rolling_days):
    # for n in range(0, 1):
        file = open(
            "Files/" + str(minute) + "ValidInequalities/" + P + "/N" + str(Num_of_SAA_samples) + "PAI" + str(
                len(PAI.index)) + "C" + str(C0) + "h" + str(h) + "beta" + str(beta) + ".txt", "a")
        Ctao = C0
        Dtk = {}
        Xtk = {}
        Mtk = {}
        M = {}
        Rtao = {}
        Gxdelay = {}
        # 定义在决策时刻，未来可能会返回的车位按平均比例0.67、 0.21、 0.12返回
        Gx = np.zeros(Num_of_periods)

        for tao in range(0, Num_of_decisions):
            # 需求Dtk
            for k in range(0, Num_of_types - int(tao * minute / 30)):
                demand.loc[tao, k] = Demand_Test.cell_value(n * Num_of_decisions + tao, k)
                Dtk[tao, k] = demand.loc[tao, k]
            print(Dtk)

            # 带入获得需求，利用SAA方法求解Xtk
            X, run, obj = Get_Xtk(int(tao * minute / 30), Ctao, demand.loc[tao, :], Gx)

            # 获得解Xtaok
            for k in range(0, min(Num_of_types, Num_of_periods - int(tao * minute / 30))):
                Xtk[tao, k] = int(X[k])

            # 获得成功匹配车位
            for k in range(0, Num_of_periods - int(tao * minute / 30)):
                Mtk[tao, k] = 0  # 当 h <= Num_of_periods - tao 时，所有 k >= K 的需求都拒绝
            for k in range(0, min(Num_of_types, Num_of_periods - int(tao * minute / 30))):
                Mtk[tao, k] = min(Xtk[tao, k], Dtk[tao, k])
            M[tao] = sum(Mtk[tao, k] for k in range(0, min(Num_of_types, Num_of_periods - int(tao * minute / 30))))
            print(Mtk)

            # 实际返回概率以及观察到返回车位数量
            if int(tao * minute / 30) == 0:
                Gxdelay[int(tao * minute / 30), 0] = sum(Mtk[e, 0] * P0 for e in range(0, tao + 1))
                Gxdelay[int(tao * minute / 30), 1] = 0
                Gxdelay[int(tao * minute / 30), 2] = 0
            elif int(tao * minute / 30) == 1:
                Gxdelay[int(tao * minute / 30), 0] = sum(
                    Mtk[e, int(tao * minute / 30) - int(e * minute / 30)] * P0 for e in range(0, tao + 1))
                Gxdelay[int(tao * minute / 30), 1] = sum(Mtk[e, 0] * P1 for e in range(0, tao + 1 - int(30 / minute)))
                Gxdelay[int(tao * minute / 30), 2] = 0
            else:
                Gxdelay[int(tao * minute / 30), 0] = sum(
                    Mtk[e, int(tao * minute / 30) - int(e * minute / 30)] * P0 for e in range(0, tao + 1))
                Gxdelay[int(tao * minute / 30), 1] = sum(
                    Mtk[e, int(tao * minute / 30) - int(e * minute / 30) - 1] * P1 for e in
                    range(0, tao + 1 - int(30 / minute)))
                Gxdelay[int(tao * minute / 30), 2] = sum(
                    Mtk[e, int(tao * minute / 30) - int(e * minute / 30) - 2] * P2 for e in
                    range(0, tao + 1 - int(60 / minute)))
            if tao%(30/minute) == 30/minute - 1:
                Gt = round(Gxdelay[int(tao * minute / 30), 0] + Gxdelay[int(tao * minute / 30), 1] + Gxdelay[
                int(tao * minute / 30), 2])
            else:
                Gt = 0

            # tao+1 时刻可用车位数量
            Ctao = Ctao - M[tao] + Gt
            print(Ctao)

            # 按比例PAI估计Rolling 过程中未返回车位
            Gx = {}
            for t in range(int((tao + 1) * minute / 30) + 1, Num_of_periods):
                G = {}
                # 当t == int(tao * minute / 30) + 1 or t == int(tao * minute / 30) + 2 时，则根据已知返回车辆利用贝叶斯调整概率
                if t == int((tao + 1) * minute / 30) + 1:
                    G[0] = sum(Mtk[e, t - int(e * minute / 30) - 1] * PAI.loc[0, t - int(e * minute / 30) - 1]
                               for e in range(0, tao + 1))
                    G[1] = sum(Mtk[e, t - int(e * minute / 30) - 2] * (1 - P0) * PAI.loc[
                        1, t - int(e * minute / 30) - 2] / (PAI.loc[1, t - int(e * minute / 30) - 2] + PAI.loc[
                        2, t - int(e * minute / 30) - 2]) for e in
                               range(0, int(tao + 1 - (tao + 1) * minute % 30)))
                    G[2] = sum(
                        Mtk[e, t - int(e * minute / 30) - 3] - Mtk[e, t - int(e * minute / 30) - 3] * P0 -
                        Mtk[e, t - int(e * minute / 30) - 3] * P1 for e in
                        range(0, int(tao + 1 - 30 / minute - (tao + 1) * minute % 30)))
                elif t == int((tao + 1) * minute / 30) + 2:
                    G[0] = sum(Mtk[e, t - int(e * minute / 30) - 1] * PAI.loc[0, t - int(e * minute / 30) - 1]
                               for e in range(0, tao + 1))
                    G[1] = sum(
                        Mtk[e, t - int(e * minute / 30) - 2] * (1 - P0) * PAI.loc[1, t - int(e * minute / 30) - 2] / (
                                PAI.loc[1, t - int(e * minute / 30) - 2] + PAI.loc[
                            2, t - int(e * minute / 30) - 2]) for e in range(0, int(tao + 1)))
                    G[2] = sum(Mtk[e, t - int(e * minute / 30) - 3] - Mtk[e, t - int(e * minute / 30) - 3] * P0 -
                               Mtk[e, t - int(e * minute / 30) - 3] * P1 for e in
                               range(0, tao + 1 - (tao + 1) * minute % 30))
                else:
                    G[0] = sum(Mtk[e, t - int(e * minute / 30) - 1] * PAI.loc[0, t - int(e * minute / 30) - 1]
                               for e in range(0, tao + 1))
                    G[1] = sum(Mtk[e, t - int(e * minute / 30) - 2] * PAI.loc[1, t - int(e * minute / 30) - 2] for e in
                               range(0, tao + 1))
                    G[2] = sum(Mtk[e, t - int(e * minute / 30) - 3] - Mtk[e, t - int(e * minute / 30) - 3] * PAI.loc[
                        0, t - int(e * minute / 30) - 3] -
                               Mtk[e, t - int(e * minute / 30) - 3] * PAI.loc[1, t - int(e * minute / 30) - 3]
                               for e in range(0, tao + 1))
                Gx[t] = round(G[0] + G[1] + G[2])
            print(Gx)
            # print(np.array(list(Gx.values())).reshape((int(tao*minute/30)+1, 23-int(tao*minute/30))))
            # 计算tao时刻收入
            Rtao[tao] = sum(
                Mtk[tao, k] * Price[int(tao * minute / 30), k] for k in
                range(0, min(Num_of_types, Num_of_periods - int(tao * minute / 30))))
            print(sum(Rtao.values()))

            # 写入求解时间
            file.write(str(n) + ',' + str(tao) + ',' + str(run) + ',' + str(obj) + ',' + str(Ctao) + ',' + str(
                Rtao[tao]) + '\n')

        # 收益
        R[n] = sum(Rtao[tao] for tao in range(0, Num_of_decisions))

        # FO
        FO[n] = sum(M[t] for t in range(0, Num_of_decisions)) \
                / sum(
            Dtk[t, k] for t in range(0, Num_of_decisions) for k in range(0, Num_of_types - int(t * minute / 30)))

        # US
        US[n] = sum(Mtk[t, k] * (k + 1) for t in range(0, Num_of_decisions) for k in
                    range(0, Num_of_types - int(t * minute / 30))) \
                / (Num_of_periods * C0)
        file.close()

    f = open("Files/" + str(minute) + "ValidInequalities/" + P + "/N" + str(Num_of_SAA_samples) + "PAI" + str(
            len(PAI.index)) + "C" + str(C0) + "h" + str(h) + "beta" + str(beta) + ".txt", "a")
    for i in range(0, Num_of_Rolling_days):
        f.write(str(R[i]) + ',' + str(FO[i]) + ',' + str(US[i]) + '\n')
    f.close()
    print(R)


if __name__ == "__main__":
    Ctao = {0: C0}
    Q = {}
    for tao in range(0, Num_of_periods):
        for j in range(0, 3):
            Q[tao, j] = 4 * j
    Rolling(cycle_time)
