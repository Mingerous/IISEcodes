import gurobipy as gp
from gurobipy import *
import numpy as np
import pandas as pd
import xlrd2
import V

P = V.P
ratio = V.ratio
C0 = 250
Num_of_Rolling_days = V.Num_of_Rolling_days
Num_of_periods = V.Num_of_periods
Num_of_types = Num_of_periods
BigM = 10000

P0 = 0.65
P1 = 0.25
P2 = 0.1

Price = V.Price

Periods = tuplelist(range(0, Num_of_periods))
Types = tuplelist(range(0, Num_of_types))
Rolling_Days = tuplelist(range(Num_of_Rolling_days))

file = open('Files/ValidInequalities/' + str(P) + '/' + str(ratio) + 'Obj' + str(C0) + '.txt', 'w')

objectives = {}
# 添加中间变量
for n in Rolling_Days:
    # 车位库存数量
    C = {0: C0}

    # N次抽样需求
    datasecond24 = xlrd2.open_workbook(
        'Files/Data/T' + str(Num_of_periods) + 'New' + str(ratio) + 'records' + str(Num_of_Rolling_days) + '.xls')
    Demand_Test = datasecond24.sheet_by_name("Newdata")
    demand = pd.DataFrame(np.zeros(Num_of_periods * Num_of_types).reshape((Num_of_periods, Num_of_types)),
                          dtype=int)

    # 建立模型
    m = gp.Model("Opt")

    # 变量
    X = m.addVars(Periods, Types, vtype=GRB.INTEGER, name='X')
    match = m.addVars(Periods, Types, vtype=GRB.INTEGER, name='match')
    B1 = m.addVars(Periods, Types, vtype=GRB.INTEGER, name='B1')
    B2 = m.addVars(Periods, Types, vtype=GRB.INTEGER, name='B2')

    # 需求
    for t in Periods:
        for k in range(0, Num_of_periods):
            if k < min(Num_of_types, Num_of_periods - t):
                demand.loc[t, k] = Demand_Test.cell_value(Num_of_periods * n + t, k)
            else:
                demand.loc[t, k] = 0
                m.addConstr(match[t, k] == 0)

    # match约束
    for t in Periods:
        for k in range(0, min(Num_of_types, Num_of_periods - t)):
            m.addConstr(match[t, k] <= X[t, k])
            m.addConstr(match[t, k] <= demand.loc[t, k])
            m.addConstr(match[t, k] >= X[t, k] - BigM * B1[t, k])
            m.addConstr(match[t, k] >= demand.loc[t, k] - BigM * B2[t, k])
            m.addConstr(B1[t, k] + B2[t, k] == 1)

    M = {}
    G = {}
    for t in Periods:
        # 库存约束
        m.addConstr(gp.quicksum(X[t, k] for k in range(0, min(Num_of_types, Num_of_periods - t))) <= C[t],
                    name="Ct")  # 预约条件中M[t, k] * P存在小数

        # 车位状态变化
        M[t] = gp.quicksum(match[t, k] for k in range(0, min(Num_of_types, Num_of_periods - t)))
        if t == 0:
            G[t] = gp.quicksum(match[s, t - s] * P0 for s in range(0, t + 1))
        elif t == 1:
            G[t] = gp.quicksum(match[s, t - s] * P0 for s in range(0, t + 1)) \
                   + gp.quicksum(match[s, t - s - 1] * P1 for s in range(0, t))
        else:
            G[t] = gp.quicksum(match[s, t - s] * P0 for s in range(0, t + 1)) \
                   + gp.quicksum(match[s, t - s - 1] * P1 for s in range(0, t)) \
                   + gp.quicksum(match[s, t - s - 2] * P2 for s in range(0, t - 1))
        C[t + 1] = C[t] - M[t] + G[t]

    # 目标函数
    m.setObjective(gp.quicksum(match[t, k] * Price[t, k] for t in Periods for k in range(0, Num_of_types)),
                   GRB.MAXIMIZE)

    # 模型求解
    m.write('Files/Opj.lp')
    m.Params.MIPFocus = 3
    m.Params.Method = 1
    m.Params.timeLimit = 300
    m.optimize()
    # if m.status == GRB.OPTIMAL:
    objectives[n] = gp.quicksum(match[t, k] * Price[t, k] for t in Periods for k in range(0, Num_of_types)).getValue()
    S1 = m.getAttr('X', X)
    print(S1)

    # FO
    FO = sum(M[t] for t in range(0, Num_of_periods)) \
         / sum(
        demand.loc[t, k] for t in range(0, Num_of_periods) for k in range(0, min(Num_of_types, Num_of_periods - t)))

    # US
    US = sum(match[t, k] * (k + 1) for t in range(0, Num_of_periods) for k in
             range(0, min(Num_of_types, Num_of_periods - t))) \
         / (Num_of_periods * C0)

    file.write(str(objectives[n]) + ' ' + str(FO.getValue()) + ' ' + str(US.getValue()) + ' ' + str(m.Runtime) + '\n')
    print(objectives[n])

file.close()
