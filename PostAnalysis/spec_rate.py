import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../')
from io_ import get_coef
from scipy.io import savemat


# for 100206 Dice
#subj = 100206


def tri_area(x1, x2, y1, y2):
        area = 1 / 2 * (y1 + y2) * (abs(x1 - x2))
        return area


def AUC(Dices):
    AUC = 0

    for k in np.arange(len(Dices)-1):
        #print(k)
        x1 = Dices.iloc[k]['thr']
        y1 = Dices.iloc[k]['Dice']
        x2 = Dices.iloc[k+1]['thr']
        y2 = Dices.iloc[k+1]['Dice']
        area = tri_area(x1, x2, y1, y2)
        AUC = AUC + area

    return AUC


def AUC_spec(Dices):
    AUC = 0

    for k in np.arange(len(Dices)-1):
        #print(k)
        x1 = Dices.iloc[k]['thr']
        y1 = Dices.iloc[k]['Spec_Rate']
        x2 = Dices.iloc[k+1]['thr']
        y2 = Dices.iloc[k+1]['Spec_Rate']
        area = tri_area(x1, x2, y1, y2)
        AUC = AUC + area

    return AUC


def subj_dice_auc(t_act, t_pred, thrs):
#     t_act = df[df['subject']==subj]['task_t'].to_numpy()[0]
#     t_pred = df[df['subject']==subj]['predict_task_t'].to_numpy()[0]

    # thr = 0.05
    #thr = 0.5

    #thrs = np.arange(0.0001,1,0.0001) # 
    Dices = pd.DataFrame()
    for k, thr in enumerate(thrs):

        topN = round(thr*len(t_act))

        # 升序排列，返回索引index,取排名靠前的topN
        N_act = np.argsort(t_act)[-topN:]
        N_pred = np.argsort(t_pred)[-topN:]

        # Dcie
        Dice = 2 * len(set(N_act).intersection(set(N_pred)))/(len(N_act) + len(N_pred))
        Dices.loc[k, 'thr'] = thr
        Dices.loc[k, 'Dice'] = Dice

    # 显示Dice系数与阈值的折线图
#     plt.plot(Dices['thr'], Dices['Dice'])
#     plt.scatter(Dices['thr'], Dices['Dice'])

    # 计算曲线的AUC

    DiceAUC = AUC(Dices)
    #print('DiceAUC =',DiceAUC)
    return DiceAUC, Dices


def main():
    base_dir = "/media/shuo/MyDrive/data/HCP/BNA/Models"
    lambda_ = 2.0
    thrs = np.arange(0.0001, 1, 0.01)
    model_path = os.path.join(base_dir, "lambda{}".format(int(lambda_)))

    splits = [0, 1, 2, 3, 4]
    session = ["REST1", "REST2"]

    aucs = []

    seed_init = 2022
    iterations = 50
    for i in range(iterations):
        for half_m in [0, 1]:
            for split_m in splits:
                for session_m in session:
                    model_prefix = "lambda_%s_%s_%s_%s_gender_" % (lambda_, session_m, split_m, half_m)
                    model_file_m = '%s0_%s.pt' % (model_prefix, seed_init - i)
                    model_file_f = '%s1_%s.pt' % (model_prefix, seed_init - i)
                    coef_m = get_coef(model_file_m, model_path).reshape((1, -1))[0, 1:]
                    coef_f = get_coef(model_file_f, model_path).reshape((1, -1))[0, 1:]

                    ## add spec_rate
                    DiceAuc, Dices = subj_dice_auc(coef_m, coef_f, thrs)
                    Dices['Spec_Rate'] = 1 - Dices['Dice']
                    AUC_Spec = AUC_spec(Dices)
                    aucs.append(AUC_Spec)
                    # for j in range(iterations):
                    #     for half_f in [0, 1]:
                    #         for split_f in splits:
                    #             for session_f in session:
                    #                 model_file_f = 'lambda_%s_%s_%s_%s_gender_0_%s.pt' % (lambda_, session_f, split_f, half_f, seed_init - j)
                    #                 coef_f = get_coef(model_file_f, model_path).reshape((1, -1))[0, 1:]
                    #                 thrs = np.arange(0.0001, 1, 0.01)
                    #                 DiceAuc, Dices = subj_dice_auc(coef_m, coef_f, thrs)
                    #
                    #                 ## add spec_rate
                    #                 Dices['Spec_Rate'] = 1 - Dices['Dice']
                    #                 AUC_Spec = AUC_spec(Dices)
                    #                 aucs.append(AUC_Spec)

    outfile = {"aucs": aucs}
    savemat("dice_score_%s.mat" % lambda_, outfile)
                                    

if __name__ == '__main__':
    main()
