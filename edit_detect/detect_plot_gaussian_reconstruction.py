def plot_gaussian_reconstruction(pipeline: DDPMPipeline, real_images,
    out_dist_real_images, fake_images, sample_num: int=100, batch_size: int
    =1024, noise_scale: Union[List[float], float]=0.01, n: int=10):
    real_images_list: List[str] = list_all_images(root=real_images)
    out_dist_real_images_list: List[str] = list_all_images(root=
        out_dist_real_images)
    fake_images_list: List[str] = list_all_images(root=fake_images)
    image_lists = [real_images_list, out_dist_real_images_list,
        fake_images_list]
    x0_dis1_set_list = []
    x0_dis2_set_list = []
    for image_list in image_lists:
        x0_dis1_set = []
        x0_dis2_set = []
        print(f'image_list: {len(image_list)}')
        for image in tqdm(image_list[:n]):
            x0_dis_ls, x0_dis1, x0_dis2 = compute_gaussian_reconstruct(pipeline
                =pipeline, noise_scale=noise_scale, batch_size=batch_size,
                image=image, num=sample_num)
            print(f'Moment: {x0_dis1}, {x0_dis2}')
            print(f'eps_dis_ls: {x0_dis_ls.shape}')
            x0_dis1_set.append(x0_dis1)
            x0_dis2_set.append(x0_dis2)
        x0_dis1_set_list.append(torch.stack(x0_dis1_set).squeeze())
        x0_dis2_set_list.append(torch.stack(x0_dis2_set).squeeze())
    ns_str: str = str(noise_scale).replace(',', '').replace('[', '').replace(
        ']', '')
    plot_scatter(x_set_list=x0_dis1_set_list, y_set_list=x0_dis2_set_list,
        fig_name=f'scatter_shift_ns{ns_str}_n{n}.jpg', title=
        f'Flow-In Rate at Timestep {noise_scale}')
