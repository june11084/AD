import vae_models
from config import *

lstm_input_size = args.chunk_len-1
h1 = 5
num_layers = 4
learning_rate = 1e-3
model = vae_models.lstm_VAE(lstm_input_size, h1, batch_size=args.batch_size, output_dim=output_dim, num_layers=num_layers)
model = model.to(device)
print(model)

loss_function = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    model.hidden = model.init_hidden()
    train_loss = 0
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        data_exclude = data[:,:,:-output_dim] # exclude the last point
        optimizer.zero_grad()
        y_pred = model(data_exclude)
        loss = loss_function(y_pred, data[:,:,-1])
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
    print('start testing')
    with torch.no_grad():
        for i, (data,) in enumerate(test_loader):
            data = data.to(device)
            data_exclude = data[:,:,:-output_dim]
            y_pred = model(data_exclude)
            test_loss = loss_function(y_pred, data[:,:,-1]).item()
            test_loss_total.append(test_loss)
    return test_loss_total

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def test_dropout():
    model.eval()
    model.apply(apply_dropout) # only making dropout layer in training mode
    test_loss_total = []
    sample_std_total = []
    print('start testing')
    with torch.no_grad():
        for i, (data,) in enumerate(test_loader):
            if i % 100 == 0: print("finish testing "+str(i)+" chunks")
            data = data.to(device)
            data_exclude = data[:,:,:-output_dim]
            tmp_loss = []
            for k in range(20):
                y_pred = model(data_exclude)
                tmp_loss.append(loss_function(y_pred, data[:,:,-1]).item())
            test_loss = np.mean(tmp_loss)
            test_loss_total.append(test_loss)
            sample_std = np.std(tmp_loss)
            sample_std_total.append(sample_std)
    return test_loss_total, sample_std_total

if __name__ == "__main__":
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    if args.train == 1:
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            scheduler.step()
        epoch = 0
        print("finish training")
        if args.dropout == 1:
            print("perform dropout in testing")
            test_loss, sample_std = test_dropout()
        else:
            print("perform testing")
            test_loss = test()
            sample_std = None
        print("finish testing")
        torch.save((test_loss, sample_std, model), args.check_path)

    else:
        (test_loss, sample_std, model) = torch.load(args.load_check)
        print("finish loading model and loss")

    # perform cut
    idx_anomaly, idx_normal = utils.cut(test_loss, 0.05)
    
    # plot hist and detect positions
    utils.plot_hist(test_loss)
    utils.plot_detect(test_loss, data_test, idx_anomaly, sample_std)

