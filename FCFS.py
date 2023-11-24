import numpy as np
import pandas as pd
import xlrd2
import V

P = V.P
ratio = V.ratio
C0 = 450
Num_of_Rolling_days = V.Num_of_Rolling_days
Num_of_periods = V.Num_of_periods
Num_of_types = Num_of_periods
BigM = 10000

P0 = 0.65
P1 = 0.25
P2 = 0.1

Ct = C0

Price = V.Price

datasecond24 = xlrd2.open_workbook('Files/Data/T' + str(Num_of_periods) + 'New' + str(ratio) + 'records' + str(Num_of_Rolling_days) + '.xls')
Demand_Test = datasecond24.sheet_by_name("Newdata")
demand = pd.DataFrame(np.zeros(Num_of_periods ** 2).reshape((Num_of_periods, Num_of_periods)), dtype=int)

def FCFS():
    objectives = {}
    FO = {}
    US = {}
    for n in range(0, Num_of_Rolling_days):
        Ct = C0
        R = 0
        M = pd.DataFrame(np.zeros(Num_of_periods ** 2).reshape((Num_of_periods, Num_of_periods)), dtype=int)
        for t in range(0, Num_of_periods):
            print(n, t, Ct)
            for k in range(0, min(Num_of_types, Num_of_periods - t)):
                demand.loc[t, k] = Demand_Test.cell_value(n * Num_of_periods + t, k)
            Dt = sum(demand.loc[t, k] for k in range(0, min(Num_of_types, Num_of_periods - t)))
            # 按比例分配表示停车车辆随机到达
            if Ct >= Dt:
                for k in range(0, min(Num_of_types, Num_of_periods - t)):
                    M.loc[t, k] = demand.loc[t, k]
            else:
                for k in range(0, min(Num_of_types, Num_of_periods - t)):
                    M.loc[t, k] = int(Ct * demand.loc[t, k] / Dt)
            Mt = sum(M.loc[t, k]for k in range(0, min(Num_of_types, Num_of_periods - t)))

            Rt = sum(M.loc[t, k] * Price[t, k] for k in range(0, min(Num_of_types, Num_of_periods - t)))

            if t == 0:
                Gt = int(sum(M.loc[e, t - e] * P0 for e in range(0, t + 1)))
            elif t == 1:
                Gt = int(sum(M.loc[e, t - e] * P0 for e in range(0, t + 1)) \
                           + sum(M.loc[e, t - e - 1] * P1 for e in range(0, t)))
            else:
                Gt = int(sum(M.loc[e, t - e] * P0 for e in range(0, t + 1)) \
                         + sum(M.loc[e, t - e - 1] * P1 for e in range(0, t)) \
                         + sum(M.loc[e, t - e - 2] * P2 for e in range(0, t - 1)))

            # print('D', Dt, 'M:', Mt, 'G:', Gt)
            print(M)
            print(Gt)
            R += Rt
            # print('R', Rt)
            Ct = Ct - Mt + Gt

        objectives[n] = R
        print('R', n, R)
        print(' ')
        # FO
        FO[n] = sum(M.loc[t, k] for t in range(0, Num_of_periods) for k in range(0, min(Num_of_types, Num_of_periods - t)))\
             / sum(demand.loc[t, k] for t in range(0, Num_of_periods) for k in range(0, min(Num_of_types, Num_of_periods - t)))

        # US
        US[n] = sum(M.loc[t, k] * (k + 1) for t in range(0, Num_of_periods) for k in
                 range(0, min(Num_of_types, Num_of_periods - t))) \
             / (Num_of_periods * C0)
    return objectives, FO, US


if __name__ == "__main__":
    file = open('Files/30ValidInequalities/' + str(P) + '/' + str(ratio) + 'FCFS' + str(C0) + '.txt', 'w')
    Obj, fo, us = FCFS()
    for n in range(0, Num_of_Rolling_days):
        print(Obj[n])
        file.write(str(Obj[n]) + ' ' + str(fo[n]) + ' ' + str(us[n]) + '\n')
    file.close()