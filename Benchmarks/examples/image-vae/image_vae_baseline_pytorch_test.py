def test(epoch, args):
    model.eval()
    losses = AverageMeter()
    test_loss = 0
    with torch.no_grad():
        for i, (_, data, _) in enumerate(val_loader_food):
            data = data.float().cuda()
            recon_batch, mu, logvar, _ = model(data)
            print('recon', recon_batch.shape, mu.shape, logvar.shape, data.
                shape)
            loss2 = loss_picture(recon_batch, data, mu, logvar, epoch)
            loss2 = torch.sum(loss2)
            losses.update(loss2.item(), int(data.shape[0]))
            test_loss += loss2.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(
                    get_batch_size(epoch, args), 3, 256, 256)[:n]])
                save_image(comparison.cpu(), output_dir + 'reconstruction_' +
                    str(epoch) + '.png', nrow=n)
                del recon_batch
                n_image_gen = 10
                images = []
                n_samples_linspace = 20
                print(data.shape)
                if data_para:
                    data_latent = model.module.encode_latent_(data[:25, ...])
                else:
                    data_latent = model.encode_latent_(data)
                print(data_latent.shape)
                print(data.shape)
                for i in range(n_image_gen):
                    pt_1 = data_latent[i * 2, ...].cpu().numpy()
                    pt_2 = data_latent[i * 2 + 1, ...].cpu().numpy()
                    sample_vec = interpolate_points(pt_1, pt_2, np.linspace
                        (0, 1, num=n_samples_linspace, endpoint=True))
                    sample_vec = torch.from_numpy(sample_vec).to(device)
                    if data_para:
                        images.append(model.module.decode(sample_vec).cpu())
                    else:
                        images.append(model.decode(sample_vec).cpu())
                save_image(torch.cat(images), output_dir + 'linspace_' +
                    str(epoch) + '.png', nrow=n_samples_linspace)
    test_loss /= len(val_loader_food.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('loss', losses.avg)
    val_losses.append(test_loss)
