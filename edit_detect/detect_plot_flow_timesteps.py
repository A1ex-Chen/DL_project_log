def plot_flow_timesteps(pipeline: DDPMPipeline, real_images,
    out_dist_real_images, fake_images, sample_num: int=100, batch_size: int
    =1024, timesteps: List[int]=[1, 2, 4, 8, 16, 32, 50, 100, 200, 400, 800]):
    real_images_list: List[str] = list_all_images(root=real_images)
    out_dist_real_images_list: List[str] = list_all_images(root=
        out_dist_real_images)
    fake_images_list: List[str] = list_all_images(root=fake_images)
    image_lists = [real_images_list, out_dist_real_images_list,
        fake_images_list]
    for timestep in timesteps:
        eps_dis1_set_list = []
        eps_dis2_set_list = []
        for image_list in image_lists:
            eps_dis1_set = []
            eps_dis2_set = []
            print(f'image_list: {len(image_list)}')
            for image in tqdm(image_list[:100]):
                eps_dis_ls, eps_dis1, eps_dis2 = compute_flow_timestep(pipeline
                    =pipeline, timestep=timestep, batch_size=batch_size,
                    image=image, num=sample_num)
                print(f'Moment: {eps_dis1}, {eps_dis2}')
                print(f'eps_dis_ls: {eps_dis_ls.shape}')
                eps_dis1_set.append(eps_dis1)
                eps_dis2_set.append(eps_dis2)
            eps_dis1_set_list.append(torch.stack(eps_dis1_set).squeeze())
            eps_dis2_set_list.append(torch.stack(eps_dis2_set).squeeze())
        plot_scatter(x_set_list=eps_dis1_set_list, y_set_list=
            eps_dis2_set_list, fig_name=f'scatter_{timestep}.jpg', title=
            f'Flow-In Rate at Timestep {timestep}')
