import glob
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

if __name__ == '__main__':
    home = str(Path.home())
    system = platform.system()
    curr_dir = os.getcwd()

    if not os.path.exists(os.path.join(curr_dir, 'summary.csv')):
        spliter = '\\' if system == 'Windows' else '/'
        fs = glob.glob(os.path.join(curr_dir, '*', '*', 'best.pth'))
        sorted(fs)
        df = pd.DataFrame(columns=['exp_name', 'commit', 'algo', 'game', 'mean', 'std', 'max', 'min', 'size', 'frames'])
        print("Reading data")
        for i, f in enumerate(tqdm(fs)):
            data = torch.load(f, map_location=torch.device('cpu'))
            rs = np.array(data['ITRs'])
            frames = data['frame_count']
            exp_name = f.split('\\')[-3]
            with open(f[:-8] + 'params.json', 'r') as fp:
                djson = json.load(fp)
                game = djson['game']
                algo = djson['algo']
                commit = djson['sha'][:6]
            df.loc[i] = [exp_name, commit, algo, game, rs.mean(), rs.std(), rs.max(), rs.min(), rs.size, frames]
        df.to_csv('summary.csv')
    else:
        df = pd.read_csv(os.path.join(curr_dir, 'summary.csv'))

    print(df)
    games = [g for g in df['game'].unique() if g != 'Pong' and g != 'Asterix']
    new_rows = []
    for i, game in enumerate(games):
        scores = df[df['game'] == game].sort_values(['mean'], ascending=False).reset_index()
        new_row = {'game': game}
        for index, row in scores.iterrows():
            new_name = row['exp_name'] + '_' + row['algo']
            new_row[new_name] = index
        new_rows.append(new_row)

    df_rank = pd.DataFrame(new_rows)
    mean_rank = {'game': 'Avg'}
    for col in df_rank.columns:
        if col != 'game':
            mean_rank[col] = df_rank[col].values.mean()

    # for row in new_rows[1:]:
    #     df_rank.append(row, ignore_index=True)
    df_rank = df_rank.append(mean_rank, ignore_index=True)
    df_rank.to_csv('rank.csv')
    print(df_rank)
