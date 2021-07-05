import numpy as np
import pandas as pd
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt


class VISUALIZATION(object):

    def __init__(self, ):

        self.eval_dir = 'eval'
        self.columns = ['inst', 'sdr', 'sir', 'sar', 'model']
        self.eval_metrics = ['sdr', 'sir', 'sar']
        self.model_name_list = ['unet', 'unet_adam_1', 'unet_adam_256', 'unet']
        self.eval_file_lists = [[os.path.join(self.eval_dir, model, file) for file in os.listdir(os.path.join(self.eval_dir, model))] for model in self.model_name_list]

    def eval_model(self, model_name, eval_files):
        out = []
        for file in eval_files:
            with open(file, 'r') as f:
                song = json.load(f)
                for inst, rst in song.items():
                    out.append([inst, rst['sdr'], rst['sir'], rst['sar'], model_name])
        return out

    def eval_models(self, ):
        rst = []
        for idx, eval_files in enumerate(self.eval_file_lists):
            rst.append(self.eval_model(self.model_name_list[idx], eval_files))
        return rst

    def get_eval_df(self, save=False):
        rst = self.eval_models()
        rst = [pd.DataFrame(item, columns=self.columns) for item in rst]
        eval_df = pd.concat(rst)
        if save: eval_df.to_csv('eval.csv', index=False)
        return eval_df


    def plot(self,):
        f, ax = plt.subplots(3, 1, figsize=(8, 6))
        eval_df = self.get_eval_df()
        for idx, metric in enumerate(self.eval_metrics):
            sns.boxplot(x='inst', y=metric, data=eval_df, ax=ax[idx], hue='model', saturation=0.4)
            # sns.swarmplot(x='inst', y=metric, data=eval_df, ax=ax[idx], hue='model')
            # sns.stripplot(x='inst', y=metric, data=eval_df, ax=ax[idx], hue='model')

        plt.tight_layout()
        plt.show()

# col_mean_separate = eval_df_separate[eval_metrics].mean()
# col_mean_no_separate = eval_df_no_separate[eval_metrics].mean()

if __name__ == '__main__':
    
    vis = VISUALIZATION()
    eval_df = vis.get_eval_df()
    vis.plot()

    print(eval_df.groupby(['model', 'inst']).median())



