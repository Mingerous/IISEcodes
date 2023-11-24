import xlrd2
from gurobipy import *
import gurobipy as gp
import numpy as np
import pandas as pd
import V


ratio = 3
C0 = 450
Num_of_Rolling_days = 21
Num_of_SAA_samples = 35
Num_of_periods = 24
Num_of_types = Num_of_periods
beta = 0.1
h = 24

P = V.P
Price = V.Price

# 假设Rolling 过程中所有被占用车位 Gx 按比例 0.65、 0.25、 0.1 返回
P0 = 0.65
P1 = 0.25
P2 = 0.1

PAI = V.PAI


def Get_Xtk():
    BigM = 10000
    N = Num_of_SAA_samples
    T = Num_of_periods
    K = Num_of_types
    Xperiods = tuplelist(range(0, T))
    Xtypes = tuplelist(range(0, K))
    Samples = tuplelist(range(0, 2))

    datafirst24 = xlrd2.open_workbook('Files/Data/24New' + str(ratio) + 'records' + str(Num_of_SAA_samples) + '.xls')
    Demand_Sample = datafirst24.sheet_by_name("Newdata")

    # 建立模型
    model = gp.Model("Get_XandY")

    # X变量
    X = model.addVars(Xperiods, Xtypes, vtype=GRB.INTEGER, name='X')
    Mx = model.addVars(Samples, Xperiods, Xtypes, vtype=GRB.INTEGER, name='Mx')
    B1 = model.addVars(Samples, Xperiods, Xtypes, vtype=GRB.BINARY, name='B1')
    B2 = model.addVars(Samples, Xperiods, Xtypes, vtype=GRB.BINARY, name='B2')
    Z = model.addVars(Samples, vtype=GRB.BINARY, name='Z')

    # N次抽样需求
    demand = tupledict()
    for n in Samples:
        for t in Xperiods:
            for k in range(0, min(K, T - t)):
                demand[n, t, k] = round(Demand_Sample.cell_value(T * n + t, k))
    # Mx约束
    for n in Samples:
        for t in Xperiods:
            for k in Xtypes:
                if k in range(0, min(K, T - t)):
                    model.addConstr(Mx[n, t, k] <= X[t, k], name=str(n)+str(t)+str(k))
                    model.addConstr(Mx[n, t, k] <= demand[n, t, k], name=str(n)+str(t)+str(k))
                    model.addConstr(Mx[n, t, k] >= X[t, k] - BigM * B1[n, t, k], name=str(n)+str(t)+str(k))
                    model.addConstr(Mx[n, t, k] >= demand[n, t, k] - BigM * B2[n, t, k], name=str(n)+str(t)+str(k))
                    model.addConstr(B1[n, t, k] + B2[n, t, k] == 1, name=str(n)+str(t)+str(k))
                else:
                    model.addConstr(X[t, k] == 0, name=str(n)+str(t)+str(k))
                    model.addConstr(Mx[n, t, k] == 0, name=str(n)+str(t)+str(k))


    # 车位状态转移
    M = {}
    C = {}
    Gx = {}
    Gxdelay = {}
    for n in Samples:
        for t in Xperiods:
            # 返回车位数量
            if t == 0:
                Gxdelay[n, t, 0] = Mx[n, 0, 0] * PAI.loc[0, 0]
                Gxdelay[n, t, 1] = 0
                Gxdelay[n, t, 2] = 0
            elif t == 1:
                Gxdelay[n, t, 0] = gp.quicksum(Mx[n, e, t - e] * PAI.loc[0, t - e] for e in range(0, t + 1))
                Gxdelay[n, t, 1] = Mx[n, 0, 0] * PAI.loc[1, 0]
                Gxdelay[n, t, 2] = 0
            elif t == 2:
                Gxdelay[n, t, 0] = gp.quicksum(Mx[n, e, t - e] * PAI.loc[0, t - e] for e in range(0, t + 1))
                Gxdelay[n, t, 1] = gp.quicksum(Mx[n, e, t - e - 1] * PAI.loc[1, t - e - 1] for e in range(0, t))
                Gxdelay[n, t, 2] = Mx[n, 0, 0] * PAI.loc[2, 0]
            else:
                Gxdelay[n, t, 0] = gp.quicksum(Mx[n, e, t - e] * PAI.loc[0, t - e] for e in range(0, t + 1))
                Gxdelay[n, t, 1] = gp.quicksum(Mx[n, e, t - e - 1] * PAI.loc[1, t - e - 1] for e in range(0, t))
                Gxdelay[n, t, 2] = gp.quicksum(Mx[n, e, t - e - 2] * PAI.loc[2, t - e - 2] for e in range(0, t - 1))
            Gx[n, t] = Gxdelay[n, t, 0] + Gxdelay[n, t, 1] + Gxdelay[n, t, 2]


            # 匹配数量以及库存数量
            M[n, t] = gp.quicksum(Mx[n, s, k] for s in range(0, t) for k in range(0, T - s))
            C[n, t] = C0 - M[n, t] + gp.quicksum(Gx[n, s] for s in range(0, t))

            # 库存约束
            model.addConstr(X.sum(t, tuplelist(range(0, min(K, T - t)))) <= C[n, t], name="c")

    # 机会约束
    model.addConstr(gp.quicksum(Z[n] for n in Samples) <= int(N * beta), name="e")

    model.update()

    # 目标函数
    ERt = gp.quicksum(Mx[n, t, k] * Price[t, k] for n in Samples for t in Xperiods for k in range(0, min(K, T - t))) / N
    ERGt = gp.quicksum(Gxdelay[n, t, j] * Q[t, j] for n in Samples for t in Xperiods for j in range(0, 3)) / N
    model.setObjective(ERt + ERGt, GRB.MAXIMIZE)

    # 模型求解参数
    model.Params.MIPFocus = 3
    model.Params.Method = 1
    model.Params.timeLimit = 3600
    model.write("Files/LP/SAA.lp")

    model.optimize()

    # 打印结果
    # if model.status == GRB.OPTIMAL:
    print(model.objVal)
    solution1 = model.getAttr('X', X)
    Z = model.getAttr('X', Z)
    for i in Samples:
        if Z[i] == 1:
            print('Z')
            print(i)
    runtime = model.Runtime
    return solution1, runtime, model.objVal


# SAA+adj
def EnhancedSAARolling():
    file = open("Files/SAA/N" + str(Num_of_SAA_samples) + P + "EnhancedSAA-C" + str(C0) + "beta" + str(beta) + ".txt", "w")
    # 带入获得需求，利用SAA方法求解Xtk
    Xtk, run, obj = Get_Xtk()
    # 解 Xtk
    X = pd.DataFrame(np.zeros(Num_of_periods))
    for tao in range(0, Num_of_periods):
        for k in range(0, min(Num_of_types, Num_of_periods - tao)):
            X.loc[tao, k] = int(Xtk[tao, k])
            file.write(str(int(X.loc[tao, k])) + ' ')
        file.write('\n')
    print(X)
    file.write('Runtime:' + str(run) + '\n')
    # 实验集需求
    Dtk = {}
    datasecond24 = xlrd2.open_workbook('Files/Data/T24New' + str(ratio) + 'records' + str(Num_of_Rolling_days) + '.xls')
    Demand_Test = datasecond24.sheet_by_name("Newdata")
    # 需求Dtk
    for n in range(0, Num_of_Rolling_days):
        for tao in range(0, Num_of_periods):
            for k in range(0, min(Num_of_types, Num_of_periods - tao)):
                Dtk[n, tao, k] = Demand_Test.cell_value(n * Num_of_periods + tao, k)

    R = {}
    FO = {}
    US = {}
    for n in range(0, Num_of_Rolling_days):
        Utk = {}
        Ntk = {}
        for tao in range(0, Num_of_periods):
            for k in range(0, min(Num_of_types, Num_of_periods - tao)):
                Utk[tao, k] = 0
                Ntk[tao, k] = 0
        Ctao = C0
        Mtk = {}
        M = {}
        Rtao = np.zeros(Num_of_periods)
        Gxdelay = {}

        # 定义在决策时刻，未来可能会返回的车位按平均比例0.65、 0.25、 0.1返回
        Gx = np.zeros(Num_of_periods)
        for tao in range(0, Num_of_periods):
            print('')
            print('tao:', tao)
            if sum(Xtk[tao, k] for k in range(0, min(Num_of_types, Num_of_periods - tao))) >= Ctao:
                for k in range(0, min(Num_of_types, Num_of_periods - tao)):
                    Xtaok = Xtk[tao, k]
                    Xtk[tao, k] = round(Ctao * Xtaok / sum(Xtk[tao, k] for k in range(0, min(Num_of_types, Num_of_periods - tao))))

            # 获得成功匹配车位
            for k in range(0, Num_of_periods - tao):
                Mtk[tao, k] = 0  # 当 h <= Num_of_periods - tao 时，所有 k >= K 的需求都拒绝
            for k in range(0, min(Num_of_types, Num_of_periods - tao)):
                Utk[tao, k] = max(0, Xtk[tao, k] - Dtk[n, tao, k])
                Ntk[tao, k] = max(0, Dtk[n, tao, k] - Xtk[tao, k])
            for j in range(0, min(Num_of_types, Num_of_periods - tao)):
                k = min(Num_of_types, Num_of_periods - tao) - j - 1
                if Ntk[tao, k] > 0:
                    for s in range(0, tao):
                        for ks in range(k + tao - s, min(Num_of_types, Num_of_periods - s)):
                            if Utk[s, ks] > 0:
                                Ntk[tao, k] = Ntk[tao, k] - min(Ntk[tao, k], Utk[s, ks])
                                Utk[s, ks] = Utk[s, ks] - min(Ntk[tao, k], Utk[s, ks])
                                if s + ks - tao - k > 0:
                                    Utk[tao+k, s+ks-tao-k] = Utk[tao+k, s+ks-tao-k] + min(Ntk[tao, k], Utk[s, ks])
                Mtk[tao, j] = Dtk[n, tao, j] - Ntk[tao, j]
            M[tao] = sum(Mtk[tao, k] for k in range(0, min(Num_of_types, Num_of_periods - tao)))

            # 实际返回概率以及观察到返回车位数量
            if tao == 0:
                Gxdelay[tao, 0] = sum(Mtk[e, tao - e] * P0 for e in range(0, tao + 1))
                Gxdelay[tao, 1] = 0
                Gxdelay[tao, 2] = 0
            elif tao == 1:
                Gxdelay[tao, 0] = sum(Mtk[e, tao - e] * P0 for e in range(0, tao + 1))
                Gxdelay[tao, 1] = sum(Mtk[e, tao - e - 1] * P1 for e in range(0, tao))
                Gxdelay[tao, 2] = 0
            else:
                Gxdelay[tao, 0] = sum(Mtk[e, tao - e] * P0 for e in range(0, tao + 1))
                Gxdelay[tao, 1] = sum(Mtk[e, tao - e - 1] * P1 for e in range(0, tao))
                Gxdelay[tao, 2] = sum(Mtk[e, tao - e - 2] * P2 for e in range(0, tao - 1))
            Gtao = round(Gxdelay[tao, 0] + Gxdelay[tao, 1] + Gxdelay[tao, 2])

            # tao+1 时刻可用车位数量
            Ctao = Ctao - M[tao] + Gtao

            # 计算tao时刻收入
            Rtao[tao] = sum(Mtk[tao, k] * Price[tao, k] for k in range(0, min(Num_of_types, Num_of_periods - tao)))

            print(Mtk)
            print(Gtao)
            print(Gx)

            # 写入库存车位和收入
            file.write(str(n) + ',' + str(tao) + ',' + str(Ctao) + ',' + str(Rtao[tao]) + '\n')


        # FO
        FO[n] = sum(M[t] for t in range(0, Num_of_periods)) \
             / sum(Dtk[n, t, k] for t in range(0, Num_of_periods) for k in
                   range(0, min(Num_of_types, Num_of_periods - t)))

        # US
        US[n] = sum(Mtk[t, k] * (k + 1) for t in range(0, Num_of_periods) for k in
                 range(0, min(Num_of_types, Num_of_periods - t))) \
             / (Num_of_periods * C0)
        # 收益
        R[n] = sum(Rtao[tao] for tao in range(0, Num_of_periods))


    for i in range(0, Num_of_Rolling_days):
        file.write(str(R[i]) + ' ' + str(FO[i]) + ' ' + str(US[i]) + '\n')
    file.close()
    print(R)


#standard  SAA
def SAARolling():
    file = open("Files/SAA/N" + str(Num_of_SAA_samples) + P + "SAA-C" + str(C0) + "beta" + str(beta) + ".txt", "w")

    # 解 Xtk
    Xtk = {}
    f = open("Files/SAA/N" + str(Num_of_SAA_samples) + P + "EnhancedSAA-C" + str(C0) + "beta" + str(beta) + ".txt", "r")
    lines = f.readlines()
    i = 0
    for line in lines:
        if i < Num_of_periods:
            l = line.strip().split(' ')
            j = 0
            for item in l:
                Xtk[i, j] = int(item)
                j += 1
            i += 1
        else:
            break

    # 实验集需求
    Dtk = {}
    datasecond24 = xlrd2.open_workbook('Files/Data/T24New' + str(ratio) + 'records' + str(Num_of_Rolling_days) + '.xls')
    Demand_Test = datasecond24.sheet_by_name("Newdata")
    # 需求Dtk
    for n in range(0, Num_of_Rolling_days):
        for tao in range(0, Num_of_periods):
            for k in range(0, min(Num_of_types, Num_of_periods - tao)):
                Dtk[n, tao, k] = Demand_Test.cell_value(n * Num_of_periods + tao, k)

    R = {}
    FO = {}
    US = {}
    Stop = {}
    for n in range(0, Num_of_Rolling_days):
        Ctao = C0
        Mtk = {}
        M = {}
        Rtao = np.zeros(Num_of_periods)
        Gxdelay = {}

        # 定义在决策时刻，未来可能会返回的车位按平均比例0.65、 0.25、 0.1返回
        Gx = np.zeros(Num_of_periods)
        for tao in range(0, Num_of_periods):
            print('')
            print('tao:', tao)
            if sum(Xtk[tao, k] for k in range(0, min(Num_of_types, Num_of_periods - tao))) >= Ctao:
                Stop[n] = tao
                break
            elif tao == 23:
                Stop[n] = 24

            # 获得成功匹配车位
            for k in range(0, Num_of_periods - tao):
                Mtk[tao, k] = 0  # 当 h <= Num_of_periods - tao 时，所有 k >= K 的需求都拒绝
            for k in range(0, min(Num_of_types, Num_of_periods - tao)):
                Mtk[tao, k] = min(Xtk[tao, k], Dtk[n, tao, k])
            M[tao] = sum(Mtk[tao, k] for k in range(0, min(Num_of_types, Num_of_periods - tao)))

            # 实际返回概率以及观察到返回车位数量
            if tao == 0:
                Gxdelay[tao, 0] = sum(Mtk[e, tao - e] * P0 for e in range(0, tao + 1))
                Gxdelay[tao, 1] = 0
                Gxdelay[tao, 2] = 0
            elif tao == 1:
                Gxdelay[tao, 0] = sum(Mtk[e, tao - e] * P0 for e in range(0, tao + 1))
                Gxdelay[tao, 1] = sum(Mtk[e, tao - e - 1] * P1 for e in range(0, tao))
                Gxdelay[tao, 2] = 0
            else:
                Gxdelay[tao, 0] = sum(Mtk[e, tao - e] * P0 for e in range(0, tao + 1))
                Gxdelay[tao, 1] = sum(Mtk[e, tao - e - 1] * P1 for e in range(0, tao))
                Gxdelay[tao, 2] = sum(Mtk[e, tao - e - 2] * P2 for e in range(0, tao - 1))
            Gtao = round(Gxdelay[tao, 0] + Gxdelay[tao, 1] + Gxdelay[tao, 2])

            # tao+1 时刻可用车位数量
            Ctao = Ctao - M[tao] + Gtao

            # 计算tao时刻收入
            Rtao[tao] = sum(Mtk[tao, k] * Price[tao, k] for k in range(0, min(Num_of_types, Num_of_periods - tao)))

            print(Mtk)
            print(Gtao)
            print(Gx)

            # 写入库存车位和收入
            file.write(str(n) + ',' + str(tao) + ',' + str(Ctao) + ',' + str(Rtao[tao]) + '\n')


        # FO
        FO[n] = sum(M[t] for t in range(0, Stop[n])) \
             / sum(Dtk[n, t, k] for t in range(0, Stop[n]) for k in
                   range(0, min(Num_of_types, Num_of_periods - t)))

        # US
        US[n] = sum(Mtk[t, k] * (k + 1) for t in range(0, Stop[n]) for k in
                 range(0, min(Num_of_types, Num_of_periods - t))) \
             / (Num_of_periods * C0)
        # 收益
        R[n] = sum(Rtao[tao] for tao in range(0, Stop[n]))


    for i in range(0, Num_of_Rolling_days):
        file.write(str(R[i]) + ' ' + str(FO[i]) + ' ' + str(US[i]) + '\n')
    file.close()
    print(R)


if __name__ == "__main__":
    Ctao = {0: C0}
    Q = {}
    for tao in range(0, Num_of_periods):
        for j in range(0, 3):
            Q[tao, j] = 2 * j

    EnhancedSAARolling()
    # SAARolling()