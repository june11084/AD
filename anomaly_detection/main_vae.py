import models
from config import *

# model = models.linear_VAE(input_size).to(device)
nz = 5
model = models.conv_VAE(nz).to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_size), reduction='sum')
    alpha = 0.5
    BCE = (1-alpha)*F.mse_loss(recon_x[:,:,:-1], x[:,:,:-1]) + alpha * F.mse_loss(recon_x[:,:,-1], x[:,:,-1]) # the recon plus the prediction (the last point)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(1e-3*BCE, 1e3*KLD)
    return 1e-3*BCE + 1e3*KLD

def pred_loss(recon_x, x):
    BCE = F.mse_loss(recon_x[:,:,-1], x[:,:,-1])
    return BCE

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        data_exclude = data[:,:,:-1] # exclude the last point
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data_exclude)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test():
    model.eval()
    test_loss_total = []
    test_mu = []
    test_logvar = []
    print('start testing')
    with torch.no_grad():
        for i, (data,) in enumerate(test_loader):
            data = data.to(device)
            # data_exclude = data[:,:,:-1]
            recon_batch, mu, logvar = model(data)
            test_mu.append(mu)
            test_logvar.append(logvar)
            # test_loss = F.mse_loss(recon_batch, data).item()
            test_loss = pred_loss(recon_batch, data).item()
            test_loss_total.append(test_loss)
            # print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss_total, test_mu, test_logvar


if __name__ == "__main__":
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)
    if args.load_check == "":
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            scheduler.step()
        epoch = 0
        print("finish training")
        test_loss, test_mu, test_logvar = test()
        x = np.array(list(map(lambda e: np.reshape(e.cpu().numpy(), (-1)), test_mu)))
        x_tsne = TSNE(n_components=2).fit_transform(x)
        torch.save((test_loss, test_mu, test_logvar, x_tsne), args.check_path)

    else:
        (test_loss, test_mu, test_logvar, x_tsne) = torch.load(args.load_check)

    # perform cut
    idx_anomaly, idx_normal = utils.cut(test_loss, 0.05)
    
    # plot hist and detect position
    utils.plot_hist(test_loss)
    utils.plot_detect(data_test, idx_anomaly)

    # get latent var
    scale = list(map(lambda e: np.reshape(e.mul(0.5).exp_().cpu().numpy(), (-1)), test_logvar)) # std = logvar.mul(0.5).exp_()
    scale = np.array(list(map(np.linalg.norm, scale)))
    scale = (scale - min(scale))/np.std(scale) ** 1.2

    # plot normal vs anomaly
    plt.scatter(x_tsne[idx_normal,0], x_tsne[idx_normal,1], s=scale[idx_normal], c='g', alpha=0.8, edgecolors='none')
    plt.scatter(x_tsne[idx_anomly,0], x_tsne[idx_anomly,1], s=scale[idx_anomly], c='r', alpha=0.8, edgecolors='none')
    plt.legend(['Normal', 'Anomaly'])
    plt.show()

    # map data time
    idx_all = list(range(len(test_loss)))
    idx_1 = idx_all[0:24]
    idx_2 = idx_all[24:48]
    idx_3 = idx_all[48:72]
    idx_4 = idx_all[72:96]
    k = 96
    while k+96 < len(idx_all):
        idx_1 += idx_all[k:k+24]
        idx_2 += idx_all[k+24:k+48]
        idx_3 += idx_all[k+48:k+72]
        idx_4 += idx_all[k+72:k+96]
        k += 96

    plt.scatter(x_tsne[idx_1,0], x_tsne[idx_1,1], s=scale[idx_1], c='b', alpha=0.8, edgecolors='none')
    plt.scatter(x_tsne[idx_2,0], x_tsne[idx_2,1], s=scale[idx_2], c='r', alpha=0.8, edgecolors='none')
    plt.scatter(x_tsne[idx_3,0], x_tsne[idx_3,1], s=scale[idx_3], c='y', alpha=0.8, edgecolors='none')
    plt.scatter(x_tsne[idx_4,0], x_tsne[idx_4,1], s=scale[idx_4], c='g', alpha=0.8, edgecolors='none')
    plt.legend(['0:00-6:00','6:00-12:00','12:00-18:00','18:00-24:00'], loc='upper right')
    plt.show()


