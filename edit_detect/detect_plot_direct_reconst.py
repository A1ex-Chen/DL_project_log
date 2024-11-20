def plot_direct_reconst(pipeline: DDPMPipeline, real_images,
    out_dist_real_images, fake_images, sample_num: int=100, batch_size: int
    =1024, timestep: int=0, n: int=10, name_prefix: str=''):
    recorder: SafetensorRecorder = SafetensorRecorder()
    real_images_list: List[str] = list_all_images(root=real_images)
    out_dist_real_images_list: List[str] = list_all_images(root=
        out_dist_real_images)
    fake_images_list: List[str] = list_all_images(root=fake_images)
    image_lists = [real_images_list, out_dist_real_images_list,
        fake_images_list]
    image_lists_labels = [0, 1, 2]
    x0_dis1_set_list = []
    x0_dis2_set_list = []
    x0_dis_trend_set_list = []
    for image_list, image_lists_label in zip(image_lists, image_lists_labels):
        x0_dis1_set = []
        x0_dis2_set = []
        x0_dis_trend_set = []
        print(f'image_list: {len(image_list)}')
        for image in tqdm(image_list[:n]):
            x0_dis_ls, x0_dis1, x0_dis2, x0_dis_trend, recorder = (
                compute_direct_reconst(pipeline=pipeline, timestep=timestep,
                batch_size=batch_size, image=image, num=sample_num,
                recorder=recorder, label=image_lists_label))
            print(f'Moment: {x0_dis1}, {x0_dis2}')
            print(f'eps_dis_ls: {x0_dis_ls.shape}')
            print(
                f'x0_dis_trend: {x0_dis_trend.shape}, x0_dis_trend_set: {len(x0_dis_trend_set)}'
                )
            x0_dis1_set.append(x0_dis1)
            x0_dis2_set.append(x0_dis2)
            x0_dis_trend_set.append(x0_dis_trend)
        x0_dis1_set_list.append(torch.stack(x0_dis1_set).squeeze())
        x0_dis2_set_list.append(torch.stack(x0_dis2_set).squeeze())
        x0_dis_trend_set_list.append(torch.stack(x0_dis_trend_set).squeeze())
    ts_str: str = str(timestep).replace(',', '').replace('[', '').replace(']',
        '')
    recorder.save(f'{name_prefix}direct_record_ts{ts_str}_n{n}', proc_mode=
        SafetensorRecorder.PROC_BEF_SAVE_MODE_STACK)
    plot_scatter(x_set_list=x0_dis1_set_list, y_set_list=x0_dis2_set_list,
        fig_name=f'{name_prefix}scatter_direct_ts{ts_str}_n{n}.jpg', title=
        f'Direct Reconstruction at Timestep {timestep}')
    plot_run(x_set_list=x0_dis_trend_set_list, fig_name=
        f'{name_prefix}line_direct_ts{ts_str}_n{n}.jpg', title=
        f'Direct Reconstruction', is_plot_var=False)
