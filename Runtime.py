import numpy as np
from textwrap import fill
import matplotlib.pyplot as plt

# Mode = 'Gurobi'
Mode = 'GurobiwithValidInequalities'

# P = 'VaryingPrice'
P = 'FlatPrice'

if P == 'VaryingPrice':
    h = 5
    maxY1 = 16
    y1 = 4
    maxY2 = 340
    y2 = 20
    suptitle1 = 'POP price'
if P == 'FlatPrice':
    h = 2
    maxY1 = 1
    y1 = 0.2
    maxY2 = 5
    y2 = 1
    suptitle1 = 'FR price'
if Mode == 'Gurobi':
    B = 'RollingHorizon'
    suptitle2 = 'SAA Model'
elif Mode == 'GurobiwithValidInequalities':
    B = 'ValidInequalities'
    suptitle2 = 'SAA Model with Valid Inequalities'


def beta():
    Vaveragetime = []
    Vmaximumtime = []
    Raveragetime = []
    Rmaximumtime = []
    X = ['0', '0.05', '0.1', '0.15']
    fig = plt.figure(figsize=(5, 3.5), dpi=180)
    plt.subplots_adjust(top=0.9, bottom=0.2, left=0.15, right=0.78, hspace=0.6, wspace=0.2)
    for b in X:
        # +有效不等式
        Vfile = open('Files/30ValidInequalities/' + P + '/N35PAI3C150H' + str(h) + 'beta' + str(b) + '.txt')
        Vlines = Vfile.readlines()
        l = 0
        Vrun = np.zeros(504)
        for line in Vlines:
            if l < 504:
                item = line.strip('\n').split(',')[2]
                Vrun[l] = float(item)
                l += 1
        Vaveragetime.append(np.sum(Vrun) / (21 * 24))
        Vmaximumtime.append(np.max(Vrun))

        # Gurobi直接求解
        Rfile = open('Files/RollingHorizon/' + P + '/N35PAI3C150H' + str(h) + 'beta' + str(b) + '.txt')
        Rlines = Rfile.readlines()
        l = 0
        Rrun = np.zeros(504)
        for line in Rlines:
            if l < 504:
                item = line.strip('\n').split(',')[2]
                Rrun[l] = float(item)
                l += 1
        Raveragetime.append(np.sum(Rrun) / (21 * 24))
        Rmaximumtime.append(np.max(Rrun))
# 平均计算时间
    X1 = [i - 0.2 for i in range(0, 4)]
    X2 = [i + 0.2 for i in range(0, 4)]
    pic = plt.subplot(1, 1, 1)
    plt.title(suptitle1, fontsize=9)
    pic.bar(X1, Vaveragetime, label=fill('SAA model with valid Inequalities', 13), width=0.4, facecolor='orange', edgecolor='white')
    pic.bar(X2, Raveragetime, label='SAA model', width=0.4, facecolor='pink', edgecolor='white')
    plt.xticks(np.arange(0, 4), ['0', '0.05', '0.1', '0.15'], fontsize=7)
    for x, y in zip(X1, Vaveragetime):
        plt.text(x, y + 0, '%.2f' % y, ha='center', va='bottom', fontsize=6)
    for x, y in zip(X2, Raveragetime):
        plt.text(x, y + 0, '%.2f' % y, ha='center', va='bottom', fontsize=6)
    plt.xlabel(r"$\beta$", fontsize=9)
    plt.ylabel('CPU Time', fontsize=9)
    plt.yticks(np.arange(0, maxY1, y1), fontsize=7)
    plt.grid(linestyle=':', color='grey', linewidth=0.3)

    if P == 'VaryingPrice':
        plt.legend(fontsize=6, bbox_to_anchor=(1, 0.2))
    plt.grid(linestyle=':', color='grey', linewidth=0.3)
    plt.tick_params(width=0.5, length=1)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    for spine in {'top', 'right', 'bottom', 'left'}:
        ax.spines[spine].set_linewidth(0.3)
    plt.show()
# 盒图
# runtime = []
# i = 1
# for beta in [0, 0.05, 0.1, 0.15]:
#     Vfile = open('Files/' + B + '/' + P + '/N35ratio3C150h' + str(h) + 'beta' + str(beta) + '.txt')
#     lines = Vfile.readlines()
#     l = 0
#     run = np.zeros(504)
#     for line in lines:
#         if l < 504:
#             item = line.strip('\n').split(',')[2]
#             run[l] = float(item)
#             l += 1
#     runtime.append(run)
#     i += 1
#     print(str(beta) + '平均计算时间：' + str(np.mean(run)))
# plt.boxplot(runtime, sym='x',vert=True)
# plt.xticks(np.arange(1, 5), ['0', '0.05', '0.1', '0.15'])
# plt.xlabel(r"$\beta$")
# plt.ylabel("CPU Time")
# plt.title(suptitle2)
# plt.show()


def C():
    fig = plt.figure(figsize=(4, 3), dpi=180)
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.1, right=0.92, hspace=0.6, wspace=1)
    i = 1
    for C0 in [150, 250, 350, 450]:
        pic = plt.subplot(1, 4, i)
        runtime = []
        Vfile1 = open('Files/RollingHorizon/' + P + '/N35PAI3C' + str(C0) + 'h' + str(h) + 'beta0.1.txt')
        lines1 = Vfile1.readlines()
        l = 0
        run1 = np.zeros(504)
        for line1 in lines1:
            if l < 504:
                item = line1.strip('\n').split(',')[2]
                run1[l] = float(item)
                l += 1
        runtime.append(run1)
        Vfile2 = open('Files/30ValidInequalities/' + P + '/N35PAI3C' + str(C0) + 'h' + str(h) + 'beta0.1.txt')
        lines2 = Vfile2.readlines()
        l = 0
        run2 = np.zeros(504)
        for line2 in lines2:
            if l < 504:
                item = line2.strip('\n').split(',')[2]
                run2[l] = float(item)
                l += 1
        runtime.append(run2)
        i += 1

        pic.boxplot(runtime, sym='.', vert=True, showfliers=False, showmeans=True, meanline=True, widths=0.5, flierprops={'markersize': 2})
        pic.tick_params(width=0.5, length=1)
        plt.ylabel('CPU Time', fontsize=5)
        plt.title(r"$C$=" + str(C0), fontsize=5)
        plt.xticks(np.arange(1, 3), [fill('SAA model', 5), fill('SAA model with valid Inequalities', 12)], fontsize=3)
        plt.yticks(fontsize=4)
        ax = plt.gca()
        ax.xaxis.set_ticks_position('bottom')
        for spine in {'top', 'right', 'bottom', 'left'}:
            ax.spines[spine].set_linewidth(0.3)

    plt.suptitle(suptitle1, fontsize=7, x=0.52, y=0.07)
    plt.show()


def N():
    runtime = []
    i = 1
    for N in [21, 35, 49]:
        Vfile = open('Files/30ValidInequalities/' + P + '/N' + str(N) + 'PAI3C150h' + str(h) + 'beta0.1.txt')
        lines = Vfile.readlines()
        l = 0
        run = np.zeros(504)
        for line in lines:
            if l < 504:
                item = line.strip('\n').split(',')[2]
                run[l] = float(item)
                l += 1
        print(np.mean(run))
        runtime.append(run)
        i += 1

    plt.boxplot(runtime, sym='x', vert=True)
    plt.xticks(np.arange(1, 4), ['21', '35', '49'])
    plt.xlabel(r"$N$")
    plt.ylabel('CPU Time')
    plt.title(suptitle1)
    plt.show()


if __name__ == "__main__":
    # beta()
    C()
    # N()