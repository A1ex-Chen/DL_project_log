# %%
import os
from typing import Union, Tuple
import pathlib
import argparse

import pandas as pd
import torch
import numpy as np

from tqdm import tqdm

from detect import plot_scatter, plot_run
from util import DirectRecorder, TensorDict, SafetensorRecorder, mse, mse_series, center_mse_series, ModelDataset, name_prefix

        


        





    
    
# def fetch_plot_fourier(x_col: str, y_col: str, output_fig_folder: str, file_name: str, real_n: int, out_dist_n: int, fake_n: int, ts: int, n: int):
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int)
    parser.add_argument('--ts', type=int, required=False)
    parser.add_argument('--o', action='store_true')

    
    args = parser.parse_args()
    print(f"n: {args.n}, ts: {args.ts}")
    
    size: int = 256
    ts: int = args.ts
    n: int = args.n
    sample_n: int = 100
    model: str = ModelDataset.MD_DDPM_EMA
    dataset: str = ModelDataset.DS_CELEBA_HQ_256
    out_dist: str = ModelDataset.OUT_DIST_FFHQ
    xscale: str='log'
    yscale: str = 'log'
    
    real_n: int = 100
    out_dist_n: int = 100
    fake_n: int = 100
    
    max_mse_mean: str = 'max_mse_mean'
    max_mse_var: str = 'max_mse_var'
    
    md: ModelDataset = ModelDataset().set_model_dataset(model=model, dataset=dataset, out_dist=out_dist)
    
    output_fig_folder: str = make_dir(f'{md.name_prefix}direct_record_fig_ts{ts}_n{n}', exist_ok=True)
    record_file: str = f'{md.name_prefix}direct_record_ts{ts}_n{n}_woSeq.safetensors'
    df_file: str = f'{md.name_prefix}direct_record_df_ts{ts}_n{n}_woSeq.hdf5'
    recorder: SafetensorRecorder = SafetensorRecorder()
    recorder.load(path=record_file, enable_update=False)
    
    # print(recorder.__data__)
    show_col_shapes(df=recorder.__data__)
    labels: torch.Tensor = recorder.__data__[SafetensorRecorder.LABEL_KEY]
    print(f"Unique Label: {labels.unique()}, Unique[0:{real_n * sample_n}]: {labels[0:real_n * sample_n].unique()}")
    print(f"Unique[{real_n * sample_n}:{(real_n + out_dist_n) * sample_n}]: {labels[real_n * sample_n:(real_n + out_dist_n) * sample_n].unique()}")
    print(f"Unique[{(real_n + out_dist_n) * sample_n}:{(real_n + out_dist_n + fake_n) * sample_n}]: {labels[(real_n + out_dist_n) * sample_n:(real_n + out_dist_n + fake_n) * sample_n].unique()}")
        
    recorder = reshape_recorder(recorder=recorder, ts=ts, size=size)
    
    print(f"After Reshape")
    show_col_shapes(df=recorder.__data__)

    if not args.o:
        df = load_cache_df(df_file)
    else:
        df = None
    
    if df is None:
        real_list = []
        out_dist_list = []
        fake_list = []
        
        images = torch.split(recorder.__data__[SafetensorRecorder.IMAGE_KEY], sample_n)
        reconsts = torch.split(recorder.__data__[SafetensorRecorder.RECONST_KEY], sample_n)
        if SafetensorRecorder.RESIDUAL_KEY in recorder.__data__:
            seqs = recorder.__data__[SafetensorRecorder.RESIDUAL_KEY]
            seqs = torch.split(seqs, sample_n)

            seq_mse_mean, seq_mse_var = compute_residual_mse(seqs=seqs, images=images, device='cuda')
        else:
            seqs = recorder.__data__[SafetensorRecorder.SEQ_KEY]
            seqs = torch.split(seqs, sample_n)
            
            seq_mse_mean, seq_mse_var = compute_seq_mse(seqs=seqs, images=images, reconsts=reconsts, device='cuda')
            seq_center_mse_mean, seq_center_mse_var = compute_seq_center_mse(seqs=seqs, images=images, reconsts=reconsts, device='cuda')
        
        if SafetensorRecorder.TRAJ_RESIDUAL_KEY in recorder.__data__:
            seqs = recorder.__data__[SafetensorRecorder.TRAJ_RESIDUAL_KEY]
            seqs = torch.split(seqs, sample_n)

            traj_mse_mean, traj_mse_var = compute_residual_mse(seqs=seqs, images=images, device='cuda')

        # seq_mse_mean: list = []
        # seq_mse_var: list = []
        # seq_center_mse_mean: list = []
        # seq_center_mse_var: list = []
        # for i, (seq, image, reconst) in tqdm(enumerate(zip(seqs, images, reconsts))):
        #     assert len(torch.unique(image, dim=0)) == 1
        #     # seq_mse_mean.append(seq.mean(dim=0).squeeze())
        #     # seq_mse_var.append(seq.var(dim=0).squeeze())
            
        #     # print(f"image: {image.shape}, seq: {seq.shape}")
        #     mse_seq = mse_series(x0=image.to('cuda'), pred_orig_images=seq.to('cuda').transpose(0, 1)).cpu()
        #     center_mse_seq = center_mse_series(pred_orig_images=seq.to('cuda').transpose(0, 1)).cpu()
            
        #     seq_mse_mean.append(mse_seq.mean(dim=0).squeeze())
        #     seq_mse_var.append(mse_seq.var(dim=0).squeeze())
            
        #     seq_center_mse_mean.append(center_mse_seq.mean(dim=0).squeeze())
        #     seq_center_mse_var.append(center_mse_seq.var(dim=0).squeeze())

        # seq_mse_mean = torch.stack(seq_mse_mean, dim=0)
        # seq_mse_var = torch.stack(seq_mse_var, dim=0)
        # seq_center_mse_mean = torch.stack(seq_center_mse_mean, dim=0)
        # seq_center_mse_var = torch.stack(seq_center_mse_var, dim=0)
        # print(f"seq_mse_var: {seq_mse_var.shape}")
        # print(f"seq_mse_var.max(dim=1)[0]: {seq_mse_var.max(dim=1)[0].shape}")
        # df: pd.DataFrame = pd.DataFrame.from_dict(seq_mse_mean)
        df: pd.DataFrame = pd.DataFrame(seq_mse_mean.numpy())
        
        # col0, col999 = df.columns[0], df.columns[999]
        # print(df[df.columns].iloc[0].shape)
        # print(f"min: {min(df[df.columns].iloc[0])}")
        
        df['max_mse_mean'] = seq_mse_mean.max(dim=1)[0]
        df['max_mse_var'] = seq_mse_var.max(dim=1)[0]
        
        df['first_mse_mean'] = seq_mse_mean[:, 0].squeeze()
        
        df['last_mse_mean'] = seq_mse_mean[:, -1].squeeze()
        df['last_mse_var'] = seq_mse_var[:, -1].squeeze()
        
        if SafetensorRecorder.SEQ_KEY in recorder.__data__:
            df['max_center_mse_mean'] = seq_center_mse_mean.max(dim=1)[0]
            df['max_center_mse_var'] = seq_center_mse_var.max(dim=1)[0]

            df['first_center_mse_mean'] = seq_center_mse_mean[:, 0].squeeze()

            df['last_center_mse_mean'] = seq_center_mse_mean[:, -1].squeeze()
            df['last_center_mse_var'] = seq_center_mse_var[:, -1].squeeze()
        
        if SafetensorRecorder.TRAJ_RESIDUAL_KEY in recorder.__data__:
            df['max_traj_mse_mean'] = traj_mse_mean.max(dim=1)[0]
            df['max_traj_mse_var'] = traj_mse_var.max(dim=1)[0]

            df['first_traj_mse_mean'] = traj_mse_mean[:, 0].squeeze()

            df['last_traj_mse_mean'] = traj_mse_mean[:, -1].squeeze()
            df['last_traj_mse_var'] = traj_mse_var[:, -1].squeeze()
            
            df['accum_traj_mse_mean'] = traj_mse_mean.cumsum(dim=1)[:, -1]
            df['accum_traj_mse_var'] = traj_mse_var.cumsum(dim=1)[:, -1]
        
        df['min'] = df[df.columns].min(axis=1)
        df['max'] = df[df.columns].max(axis=1)
        df['drop'] = df['first_mse_mean'] - df['min']
        df['increase'] = df['last_mse_mean'] - df['min']
        # df['label'] = ['Real'] * 100 + ['Out-Dist'] * 12 + ['Fake'] * 100
        
        # df.loc[:, df.columns] = df[df.columns].map(str)
        # df = df.rename(str, axis='columns')
    # else:
        # col0, col999 = df.columns[0], df.columns[999]
        # print(df[df.columns].iloc[0].shape)
        # print(f"min: {min(df[df.columns].iloc[0])}")
        
    save_cache_df(df=df, file=df_file)
    print(df)
    
    print(df['drop'])
    print(recorder.__data__[SafetensorRecorder.LABEL_KEY])
    
    # x_set_list, y_set_list = x_y_set_list_gen(df=df, x_label='drop', y_label='increase', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    # plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Drop & Increment at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'{md.name_prefix}scatter_drop_incr_ts{ts}_n{n}'), xlabel='Drop', ylabel='Increase')
    fetch_plot_scatter(df=df, md=md, x_col='drop', y_col='increase', x_axis='Drop', y_axis='Increment', output_fig_folder=output_fig_folder, file_name='scatter_drop_incr', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n, ts=ts, n=n, xscale=xscale, yscale=yscale)
    
    # x_set_list, y_set_list = x_y_set_list_gen(df=df, x_label='drop', y_label='last_mse_mean', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    # plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Drop & End at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'{md.name_prefix}scatter_drop_end_ts{ts}_n{n}'), xlabel='Drop', ylabel='End')
    fetch_plot_scatter(df=df, md=md, x_col='drop', y_col='last_mse_mean', x_axis='Drop', y_axis='End', output_fig_folder=output_fig_folder, file_name='scatter_drop_end', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n, ts=ts, n=n, xscale=xscale, yscale=yscale)
    
    # x_set_list, y_set_list = x_y_set_list_gen(df=df, x_label='drop', y_label='max_center_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    # plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Drop & Max Center MSE Var at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'{md.name_prefix}scatter_drop_max_center_mse_var_ts{ts}_n{n}'), xlabel='Drop', ylabel='Max Center MSE Var')
    
    # x_set_list, y_set_list = x_y_set_list_gen(df=df, x_label='drop', y_label='last_center_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    # plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Drop & Last Center MSE Var at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'{md.name_prefix}scatter_drop_last_center_mse_var_ts{ts}_n{n}'), xlabel='Drop', ylabel='Last Center MSE Var')
    
    # x_set_list, y_set_list = x_y_set_list_gen(df=df, x_label='last_mse_mean', y_label='last_center_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    # plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Last MSE Mean & Last Center MSE Var at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'{md.name_prefix}scatter_last_mse_mean_last_center_mse_var_ts{ts}_n{n}'), xlabel='Last MSE Mean', ylabel='Last Center MSE Var')
    
    # x_set_list, y_set_list = x_y_set_list_gen(df=df, x_label='last_mse_mean', y_label='last_center_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    # plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Last MSE Mean & Last Center MSE Var at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'{md.name_prefix}scatter_last_mse_mean_last_center_mse_var_ts{ts}_n{n}'), xlabel='Last MSE Mean', ylabel='Last Center MSE Var')
    
    fetch_plot_scatter(df=df, md=md, x_col='drop', y_col='max_mse_var', x_axis='Drop', y_axis='Max MSE Var', output_fig_folder=output_fig_folder, file_name='scatter_drop_max_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n, ts=ts, n=n, xscale=xscale, yscale=yscale)
    
    fetch_plot_scatter(df=df, md=md, x_col='drop', y_col='last_mse_var', x_axis='Drop', y_axis='Last MSE Var', output_fig_folder=output_fig_folder, file_name='scatter_drop_last_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n, ts=ts, n=n, xscale=xscale, yscale=yscale)
    
    fetch_plot_scatter(df=df, md=md, x_col='last_mse_mean', y_col='last_mse_var', x_axis='Last MSE Mean', y_axis='Last MSE Var', output_fig_folder=output_fig_folder, file_name='scatter_last_mse_mean_last_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n, ts=ts, n=n, xscale=xscale, yscale=yscale)
    
    fetch_plot_scatter(df=df, md=md, x_col='max_mse_mean', y_col='max_mse_var', x_axis='Max MSE Mean', y_axis='Max MSE Var', output_fig_folder=output_fig_folder, file_name='scatter_max_mse_mean_max_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n, ts=ts, n=n, xscale=xscale, yscale=yscale)
    
    fetch_plot_scatter(df=df, md=md, x_col='max_mse_mean', y_col='last_mse_var', x_axis='Max MSE Mean', y_axis='Last MSE Var', output_fig_folder=output_fig_folder, file_name='scatter_max_mse_mean_last_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n, ts=ts, n=n, xscale=xscale, yscale=yscale)
    
    fetch_plot_scatter(df=df, md=md, x_col='last_mse_mean', y_col='max_mse_var', x_axis='Last MSE Mean', y_axis='Max MSE Var', output_fig_folder=output_fig_folder, file_name='scatter_last_mse_mean_max_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n, ts=ts, n=n, xscale=xscale, yscale=yscale)
    
    fetch_plot_run(df=df, md=md, x_col=df.columns[:ts], y_col='max_mse_var', output_fig_folder=output_fig_folder, file_name='line_direct', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n, ts=ts, n=n)
    
    fetch_plot_scatter(df=df, md=md, x_col='last_traj_mse_mean', y_col='last_traj_mse_var', x_axis='Last Traj MSE Mean', y_axis='Last Traj MSE Var', output_fig_folder=output_fig_folder, file_name='scatter_last_traj_mse_mean_last_traj_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n, ts=ts, n=n, xscale=xscale, yscale=yscale)
    
    fetch_plot_scatter(df=df, md=md, x_col='max_traj_mse_mean', y_col='max_traj_mse_var', x_axis='Max Traj  MSE Mean', y_axis='Max Traj MSE Var', output_fig_folder=output_fig_folder, file_name='scatter_max_traj_mse_mean_max_traj_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n, ts=ts, n=n, xscale=xscale, yscale=yscale)
    
    fetch_plot_scatter(df=df, md=md, x_col='max_traj_mse_mean', y_col='last_traj_mse_var', x_axis='Max Traj  MSE Mean', y_axis='Last Traj MSE Var', output_fig_folder=output_fig_folder, file_name='scatter_max_traj_mse_mean_last_traj_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n, ts=ts, n=n, xscale=xscale, yscale=yscale)
    
    fetch_plot_scatter(df=df, md=md, x_col='last_traj_mse_mean', y_col='max_traj_mse_var', x_axis='Last Traj  MSE Mean', y_axis='Max Traj MSE Var', output_fig_folder=output_fig_folder, file_name='scatter_last_traj_mse_mean_max_traj_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n, ts=ts, n=n, xscale=xscale, yscale=yscale)
    
    fetch_plot_scatter(df=df, md=md, x_col='accum_traj_mse_mean', y_col='accum_traj_mse_var', x_axis='Accum Traj  MSE Mean', y_axis='Accum Traj MSE Var', output_fig_folder=output_fig_folder, file_name='scatter_accum_traj_mse_mean_accum_traj_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n, ts=ts, n=n, xscale=xscale, yscale=yscale)
    
    fetch_plot_scatter(df=df, md=md, x_col='accum_traj_mse_mean', y_col='first_mse_mean', x_axis='Accum Traj  MSE Mean', y_axis='First MSE Mean', output_fig_folder=output_fig_folder, file_name='scatter_accum_traj_mse_mean_first_mse_mean', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n, ts=ts, n=n, xscale=xscale, yscale=yscale)
    
    residuals: torch.Tensor = mse(x0=recorder.__data__[SafetensorRecorder.IMAGE_KEY], pred_x0=recorder.__data__[SafetensorRecorder.RECONST_KEY])
    
    
# %%