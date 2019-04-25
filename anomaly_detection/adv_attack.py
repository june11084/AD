from config import *
import foolbox

loss_function = torch.nn.MSELoss(size_average=False)

# attack
if __name__ == "__main__":
    (test_loss, model) = torch.load(args.load_check)
    from torch.autograd import Variable
    model.train()
    orig_loss_list = []
    new_loss_list = []
    data_mod_list = []
    eta = 1e-2
    for batch_idx, (data,) in enumerate(test_loader):
        data = data.to(device)
        data_exclude = data.clone() # this is necessary otherwise it will modify original data
        data_exclude = Variable(data_exclude[:,:,:-output_dim], requires_grad=True)
        loss = loss_function(model(data_exclude), data[:,:,-1])
        orig_loss_list.append(loss.item())
        y_pred_old = torch.cuda.FloatTensor([1000])
        for k in range(1000):
            y_pred = model(data_exclude)
            if torch.abs(y_pred.cpu() - y_pred_old.cpu()) < 1e-8:
                break
            y_pred_old = y_pred
            adv_loss = -loss_function(y_pred, data[:,:,-1])
            adv_loss.backward()
            data_exclude.data -= eta*data_exclude.grad.data
            for i in range(data_exclude.data.numel()):
                data_exclude.data[0,0,i].clamp_(data.data[0,0,i].item()*(1-args.attack_bound), data.data[0,0,i].item()*(1+args.attack_bound))
        data_mod_list.append(np.reshape(data_exclude.cpu().detach().numpy(), (-1)))
        new_loss = loss_function(model(data_exclude), data[:,:,-1])
        new_loss_list.append(new_loss.item())
        print(loss.item(), new_loss.item())
        if args.attack_savepath != "":
            torch.save((orig_loss_list, new_loss_list, data_mod_list), args.attack_savepath)
        # plt.plot(np.reshape(data.cpu().numpy(), (-1)), 'g')
        # new_data = np.concatenate((np.reshape(data_exclude.cpu().detach().numpy(), (-1)), np.reshape(y_pred.cpu().detach().numpy(), (-1))), axis=0)
        # plt.plot(new_data, 'r')
        # plt.ylim(0,1)
        # plt.legend([str(loss.item()), str(new_loss.item())])
        # plt.show()