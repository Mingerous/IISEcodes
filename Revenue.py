import matplotlib.pyplot as plt
import numpy as np
from textwrap import fill

b = 0.1
Num_of_Rolling_days = 21

P = 'VaryingPrice'
# P = 'FlatPrice'

if P == 'VaryingPrice':
    suptitle = 'POP price'
    C0h = 5
elif P == 'FlatPrice':
    suptitle = 'FR price'
    C0h = 2

linewith = 0.6
markersize = 1.2

def C0():
    fig = plt.figure(figsize=(4, 5), dpi=180)
    plt.suptitle(suptitle, fontsize=10, x=0.5, y=0.05)
    plt.subplots_adjust(top=0.96, bottom=0.1, left=0.12, right=0.84, hspace=0.6, wspace=0.2)
    for C0 in [150, 250, 350, 450]:
        if C0 == 150:
            pic = plt.subplot(4, 1, 1)
            if P == 'FlatPrice':
                plt.yticks(np.linspace(1000, 3000, 5), fontsize=5)
            elif P == 'VaryingPrice':
                plt.yticks(np.linspace(0, 6000, 7), fontsize=5)
        elif C0 == 250:
            pic = plt.subplot(4, 1, 2)
            if P == 'FlatPrice':
                plt.yticks(np.linspace(3000, 5500, 6), fontsize=5)
            elif P == 'VaryingPrice':
                plt.yticks(np.linspace(0, 10000, 11), fontsize=5)
        elif C0 == 350:
            pic = plt.subplot(4, 1, 3)
            if P == 'FlatPrice':
                plt.yticks(np.linspace(4500, 6500, 5), fontsize=5)
            elif P == 'VaryingPrice':
                plt.yticks(np.linspace(0, 15000, 16), fontsize=5)
        elif C0 == 450:
            pic = plt.subplot(4, 1, 4)
            if P == 'FlatPrice':
                plt.yticks(np.linspace(5000, 7000, 5), fontsize=5)
            elif P == 'VaryingPrice':
                plt.yticks(np.linspace(0, 15000, 16), fontsize=5)
        title = r"$C$=" + str(C0)

        x = np.array(range(1, Num_of_Rolling_days + 1))
        # OPT
        Objf = open('Files/30ValidInequalities/' + P + '/' + '3Obj' + str(C0) + '.txt', 'r')
        Objlines = Objf.readlines()
        Obj = np.zeros(21)
        k = 1
        for line in Objlines:
            Obj[k - 1] = line.strip('\n').split(' ')[0]
            k += 1
        print(Obj)
        pic.plot(x, Obj, marker='.', label='OPT', lw=linewith, markersize=markersize, linestyle=':')

        # Our DA
        Rollingf = open('Files/30ValidInequalities/' + P + '/N35PAI3C' + str(C0) + 'h' + str(C0h) + 'beta' + str(b) + '.txt', 'r')
        Rollinglines = Rollingf.readlines()
        Rolling = np.zeros(21)
        k = 1
        for line in Rollinglines:
            k += 1
            if k in range(506, 527):
                item = line.strip('\n').split(',')[0]
                # Rolling[k - 506] = 1 - float(item)/float(Obj[k - 506])
                Rolling[k - 506] = item
        print(Rolling)
        pic.plot(x, Rolling, marker='.', label='Our DA', lw=linewith, markersize=markersize, linestyle='-')

        # SA + ADJ
        EnhancedSAAf = open('Files/SAA/N35' + P + 'EnhancedSAA-C' + str(C0) + 'beta' + str(0.1) + '.txt', 'r')
        EnhancedSAAlines = EnhancedSAAf.readlines()
        EnhancedSAA = np.zeros(21)
        k = 1
        for line in EnhancedSAAlines:
            k += 1
            if k in range(531, 552):
                item = line.strip('\n').split(' ')[0]
                # EnhancedSAA[k - 531] = 1 - float(item)/float(Obj[k - 531])
                EnhancedSAA[k - 531] = item
        print(EnhancedSAA)
        pic.plot(x, EnhancedSAA, marker='.', label='SAA+ADJ', lw=linewith, markersize=markersize, linestyle='--')

        # FCFS
        FCFSf = open('Files/30ValidInequalities/' + P + '/3FCFS' + str(C0) + '.txt', 'r')
        FCFSlines = FCFSf.readlines()
        FCFS = np.zeros(21)
        k = 1
        for line in FCFSlines:
            item = line.strip('\n').split(' ')[0]
            # FCFS[k - 1] = 1 - float(item)/float(Obj[k - 1])
            FCFS[k - 1] = item
            k += 1
        print(FCFS)
        pic.plot(x, FCFS, marker='p', label='FCFS', lw=linewith, markersize=markersize, linestyle=':')

        # Standard SA
        SAAf = open('Files/SAA/N35' + P + 'SAA-C' + str(C0) + 'beta' + str(0.1) + '.txt', 'r')
        SAAlines = SAAf.readlines()
        length = len(SAAlines)
        SAA = np.zeros(21)
        k = 1
        for line in SAAlines:
            k += 1
            if k in range(length-19, length+2):
                item = line.strip('\n').split(' ')[0]
                # SAA[k - 317] = 1 - float(item)/float(Obj[k - length + 19])
                SAA[k - length + 19] = item
        print(SAA)
        pic.plot(x, SAA, marker='.', label=fill('Standard SAA', 10), lw=linewith, markersize=markersize, linestyle='-.')

        plt.xlabel('testing days', fontsize=6, position=(0.9, 0))
        plt.ylabel('Revenue', fontsize=6)
        plt.xticks(np.linspace(1, 21, 21), fontsize=5)
        plt.grid(linestyle=':', color='grey', linewidth=0.3)
        plt.title(title, fontsize=8, x=0.5, y=0.87)
        plt.tick_params(width=0.5, length=1)
        ax = plt.gca()
        ax.xaxis.set_ticks_position('bottom')
        for spine in {'top', 'right', 'bottom', 'left'}:
           ax.spines[spine].set_linewidth(0.3)

    if P == 'VaryingPrice':
        plt.legend(fontsize=5, ncol=1, bbox_to_anchor=(1.22, 1.18))
    plt.show()
    fig.savefig('Files/'+P+'.png', format='png', dpi=180)


def H():
    fig = plt.figure(figsize=(4, 5), dpi=180)
    plt.suptitle(suptitle, fontsize=10, x=0.5, y=0.05)
    plt.subplots_adjust(top=0.96, bottom=0.1, left=0.12, right=0.84, hspace=0.6, wspace=0.2)
    for C0 in {150, 250, 350, 450}:
        if C0 == 150:
            pic = plt.subplot(4, 1, 1)
            if P == 'FlatPrice':
                plt.yticks(np.linspace(2800, 3200, 5), fontsize=5)
            elif P == 'VaryingPrice':
                plt.yticks(np.linspace(5000, 6200, 7), fontsize=5)
        elif C0 == 250:
            pic = plt.subplot(4, 1, 2)
            if P == 'FlatPrice':
                plt.yticks(np.linspace(4600, 5500, 10), fontsize=5)
            elif P == 'VaryingPrice':
                plt.yticks(np.linspace(8000, 10000, 11), fontsize=5)
        elif C0 == 350:
            pic = plt.subplot(4, 1, 3)
            if P == 'FlatPrice':
                plt.yticks(np.linspace(5900, 6400, 6), fontsize=5)
            elif P == 'VaryingPrice':
                plt.yticks(np.linspace(10000, 12000, 6), fontsize=5)
        elif C0 == 450:
            pic = plt.subplot(4, 1, 4)
            if P == 'FlatPrice':
                plt.yticks(np.linspace(6300, 6800, 6), fontsize=5)
            elif P == 'VaryingPrice':
                plt.yticks(np.linspace(11000, 13000, 6), fontsize=5)
        title = r"$C$=" + str(C0)

        x = np.array(range(1, Num_of_Rolling_days + 1))

        #
        linestyle = ['-', '--', '-.', ':', '-']
        for h in range(1, 6):
            Rollingf = open(
                'Files/30ValidInequalities/' + P + '/N35PAI3C' + str(C0) + 'h' + str(h) + 'beta' + str(b) + '.txt', 'r')
            Rollinglines = Rollingf.readlines()
            Rolling = np.zeros(21)
            k = 1
            for line in Rollinglines:
                k += 1
                if k in range(506, 527):
                    item = line.strip('\n').split(',')[0]
                    # Rolling[k - 506] = 1 - float(item)/float(Obj[k - 506])
                    Rolling[k - 506] = item
            print(Rolling)
            pic.plot(x, Rolling, marker='.', label='h='+str(h), ls=linestyle[h-1], lw=linewith, markersize=markersize)

        plt.xlabel('testing days', fontsize=6, position=(0.9, 0))
        plt.ylabel('Revenue', fontsize=6)
        plt.xticks(np.linspace(1, 21, 21), fontsize=5)
        plt.grid(linestyle=':', color='grey', linewidth=0.3)
        plt.title(title, fontsize=8, x=0.5, y=0.9)
        plt.tick_params(width=0.5, length=1)
        ax = plt.gca()
        for spine in {'top', 'right', 'bottom', 'left'}:
            ax.spines[spine].set_linewidth(0.3)

    if P == 'VaryingPrice':
        plt.legend(fontsize=5, ncol=1, bbox_to_anchor=(1.17, -0.58))
    plt.show()
    fig.savefig('Files/' + P + '.png', format='png', dpi=180)


def beta():
    fig = plt.figure(figsize=(5, 3.5), dpi=180)
    plt.subplots_adjust(top=0.9, bottom=0.2, left=0.15, right=0.78, hspace=0.6, wspace=0.2)
    x = [i for i in range(0, 4)]

    Objf = open('Files/30ValidInequalities/' + P + '/' + '3Obj150.txt', 'r')
    Objlines = Objf.readlines()
    Obj = np.zeros(21)
    k = 1
    for line in Objlines:
        Obj[k - 1] = line.strip('\n').split(' ')[0]
        k += 1

    gap = []
    for b in [0, 0.05, 0.1, 0.15]:
        Rollingf = open('Files/30ValidInequalities/' + P + '/N35PAI3C150h' + str(C0h) + 'beta' + str(b) + '.txt', 'r')
        Rollinglines = Rollingf.readlines()
        Rolling = np.zeros(21)
        k = 1
        for line in Rollinglines:
            k += 1
            if k in range(506, 527):
                item = line.strip('\n').split(',')[0]
                # Rolling[k - 506] = 1 - float(item)/float(Obj[k - 506])
                Rolling[k - 506] = item
        gap.append((1 - np.mean(Rolling)/np.mean(Obj)))
    print(gap)
    plt.bar(x, gap, width=0.5, edgecolor='white')


    plt.title(suptitle, fontsize=9)
    plt.ylabel("Gap", fontsize=9, position=(0, 0.5))
    plt.xlabel(r"$\beta$", fontsize=9, position=(0.5, 0))
    plt.yticks(np.linspace(0, 0.08, 5), ['0', '2%', '4%', '6%', '8%'], fontsize=7)
    plt.xticks(x, ['0', '0.05', '0.1', '0.15'], fontsize=7)
    for i, j in zip(x, gap):
        plt.text(i, j, '{:.2f}%'.format(100*j), ha='center', va='bottom', fontsize=6)
    plt.grid(linestyle=':', color='grey', linewidth=0.3)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    for spine in {'top', 'right', 'bottom', 'left'}:
        ax.spines[spine].set_linewidth(0.3)
    plt.ylim((0, 0.08))
    plt.show()


def UFR():
    fig = plt.figure(figsize=(10, 3), dpi=180)
    plt.subplots_adjust(top=0.9, bottom=0.25, left=0.05, right=0.87, hspace=0.6, wspace=0.3)

    x = ['9:00-9:30', '9:30-10:00', '10:00-10:30', '10:30-11:00', '11:00-11:30', '11:30-12:00', '12:00-12:30', '12:30-13:00', '13:00-13:30', '13:30-14:00', '14:00-14:30', '14:30-15:00',
         '15:00-15:30', '15:30-16:00', '16:00-16:30', '16:30-17:00', '17:00-17:30', '17:30-18:00', '18:00-18:30', '18:30-19:00', '19:00-19:30', '19:30-20:00', '20:00-20:30', '20:30-21:00']
    flatrate = [0, 0.022380952, 0.128095238, 0.18047619, 0.176190476, 0.227619048, 0.255714286,	0.341428571, 0.39952381,
                0.446190476, 0.43, 0.421904762,	0.484761905, 0.473809524, 0.542857143, 0.643809524, 0.706666667,
                0.731428571, 0.769047619, 0.79047619, 0.738095238, 0.67, 0.5, 0]
    inventory1 = [150, 103.1904762, 50.28571429, 39.38095238, 38.42857143, 36.66666667, 31.66666667, 27, 21.19047619,
                  18.76190476, 17.47619048, 16.14285714, 16.61904762, 17.61904762, 17.0952381, 18.42857143, 21.57142857,
                  26.28571429, 29.95238095, 31.38095238, 27.38095238, 26.85714286, 25.47619048, 29.66666667]
    poprate = [0.038095238, 0.204285714, 0.406190476, 0.490952381, 0.083333333,	0.186190476, 0.19047619, 0.222380952,
               0.586190476,	0.560952381, 0.553333333, 0.495238095, 0.603333333, 0.277142857, 0.157142857, 0.167619048,
               0.068571429, 0.033333333, 0.674761905, 0.771428571, 0.738095238, 0.67, 0.5, 0]
    inventory2 = [150, 107.1904762, 61.61904762, 48.57142857, 58.76190476, 52.57142857, 40.23809524, 33.38095238,
                  12.9047619, 13.66666667, 21.57142857, 29.57142857, 34.95238095, 37.66666667, 39.14285714, 40.61904762,
                  39.9047619, 43.57142857, 70.28571429, 42.76190476, 42.47619048, 44.04761905, 19.38095238, 21.0952381]

    plt.grid(linestyle=':', color='grey', linewidth=0.3)
    pic1 = plt.subplot(1, 2, 1)
    plt.title("FR Price", fontsize=6)
    plt.xlabel('horizon', fontsize=6, position=(0.9, 0))
    plt.xticks(range(0, len(x)), fontsize=5, rotation=90)
    plt.yticks(np.linspace(0, 1, 6), fontsize=5)
    plt.ylim((0, 1))
    pic1.bar(x, flatrate, width=0.4)
    pic3 = pic1.twinx()
    plt.yticks(range(0, 161, 20), fontsize=5)
    plt.ylim((0, 160))
    pic3.plot(x, inventory1, marker='x', markersize=2, color='orange', linewidth=0.5)

    pic2 = plt.subplot(1, 2, 2)
    plt.title("POP Price", fontsize=6)
    plt.xlabel('horizon', fontsize=6, position=(0.9, 0))
    plt.xticks(range(0, len(x)), fontsize=5, rotation=90)
    plt.yticks(np.linspace(0, 1, 6), fontsize=5)
    plt.ylim((0, 1))
    pic2.bar(x, poprate, width=0.4, label='$PRO_t$')
    pic4 = pic2.twinx()
    plt.yticks(range(0, 161, 20), fontsize=5)
    plt.ylim((0, 160))
    pic4.plot(x, inventory2, marker='x', markersize=2, color='orange', linewidth=0.5, label='$C_t$')

    fig.legend(fontsize=5, ncol=1, bbox_to_anchor=(0.99, 0.35))
    plt.show()


if __name__ == "__main__":
    # C0()
    # H()
    # N()
    # beta()
    UFR()