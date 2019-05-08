import models
from config import *

# hyper params
lstm_input_size = args.chunk_len-1
h1 = 20
num_layers = 4
learning_rate = 1e-3

# build model
model = models.LSTM(lstm_input_size, h1, batch_size=args.batch_size, output_dim=output_dim, num_layers=num_layers)
model = model.to(device)

# set loss func and optimizer
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
    print('start testing student')
    with torch.no_grad():
        for i, (data,) in enumerate(test_loader):
            data = data.to(device)
            data_exclude = data[:,:,:-output_dim]
            y_pred = model(data_exclude)
            test_loss = loss_function(y_pred, data[:,:,-1]).item()
            test_loss_total.append(test_loss)
    return test_loss_total

# train the student model
def train_student(epoch):
    model_student.train()
    train_loss = 0
    alpha = args.alpha
    for batch_idx, (data, std) in enumerate(train_loader):
        data = data.to(device)
        data_exclude = data[:,:,:-output_dim] # exclude the last point
        target_logvar = torch.log(std*std).to(device) # logvar = log(std^2)
        optimizer.zero_grad()
        y_pred, logvar = model_student(data_exclude)
        loss = loss_function(y_pred, data[:,:,-1]) + alpha*loss_function(logvar, target_logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

# test the student model
def test_student():
    # batch_size needs to be 1
    alpha = args.alpha
    model_student.eval()
    test_loss_total = []
    std_total = []
    print('start testing student')
    with torch.no_grad():
        for i, (data, std) in enumerate(test_loader):
            data = data.to(device)
            data_exclude = data[:,:,:-output_dim]
            target_logvar = torch.log(std*std).to(device) # logvar = log(std^2)
            y_pred, logvar = model_student(data_exclude)
            test_loss = loss_function(y_pred, data[:,:,-1]) + alpha*loss_function(logvar, target_logvar)
            test_loss_total.append(test_loss.item())
            std_total.append(torch.sqrt(torch.exp(logvar)).item())
    return test_loss_total, std_total

# apply dropout in testing
def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

# testing with dropout
def test_dropout(dataloader):
    model.eval()
    model.apply(apply_dropout) # only making dropout layer in training mode
    test_loss_total = []
    sample_std_total = []
    print('start testing with dropout')
    with torch.no_grad():
        for i, (data,) in enumerate(dataloader):
            if i % 100 == 0: print("finish testing "+str(i)+" chunks")
            data = data.to(device)
            data_exclude = data[:,:,:-output_dim]
            tmp_loss = []
            for k in range(100):
                y_pred = model(data_exclude)
                tmp_loss.append(loss_function(y_pred, data[:,:,-1]).item())
            test_loss = np.mean(tmp_loss)
            test_loss_total.append(test_loss)
            sample_std = np.std(tmp_loss)
            sample_std_total.append(sample_std)
    return test_loss_total, sample_std_total

if __name__ == "__main__":
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)

    # train teacher model
    if args.train == 'train_teacher':
        print(model)
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            scheduler.step()
        epoch = 0
        torch.save(model, args.check_path)
        print("finish training, save model to "+args.check_path)
        

    # test
    elif args.train is 'test': 
        model = torch.load(args.load_check)
        print(model)
        if args.dropout is not None:
            print("perform dropout in testing")
            test_loss, sample_std = test_dropout(test_loader)
        else:
            print("perform testing")
            test_loss = test()
            sample_std = None
        torch.save((test_loss, sample_std, model), args.check_path)
        print("finish testing, save model to "+args.check_path)


    # # only load results
    # elif args.train == 'load':
    #     (test_loss, sample_std, model) = torch.load(args.load_check)
    #     print(model)
    #     print("finish loading model and loss from "+args.load_check)


    # train student model
    elif args.train == 'train_student':
        print("train the student model, read teacher from ", args.load_check)
        model = torch.load(args.load_check)
        model_student = models.LSTM_student(lstm_input_size, h1, batch_size=args.batch_size, output_dim=output_dim, num_layers=num_layers)
        model_student = model_student.to(device)
        optimizer = torch.optim.Adam(model_student.parameters(), lr=args.lr)
        
        print('teacher: ', model)
        print('student: ', model_student)
        print("read dataset with std from", args.custom_data)
        trainset_with_std, testset_with_std = torch.load(args.custom_data)
        train_loader = torch.utils.data.DataLoader(trainset_with_std, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset_with_std, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        for epoch in range(1, args.epochs + 1):
            train_student(epoch)
            scheduler.step()
        epoch = 0
        torch.save(model_student, args.check_path)
        print("finish training student, save model to "+args.check_path)
 
    
    # test student model
    elif args.train == 'test_student':
        print("test the student model, read student from ", args.load_check)
        model_student = torch.load(args.load_check)
        model_student = model_student.to(device)
        print('student: ', model_student)
        print("read dataset with std from", args.custom_data)
        trainset_with_std, testset_with_std = torch.load(args.custom_data)
        test_loader = torch.utils.data.DataLoader(testset_with_std, batch_size=1, num_workers=args.num_workers, shuffle=False)
        test_loss, sample_std = test_student()


    else:
        print("skip training, testing, loading")


    # create dataset with std, can only use batch size 1
    if args.build_std is not None:
        print("read teacher from ", args.load_check)
        model = torch.load(args.load_check)
        print(model)

        # obtain std
        print("start dropout testing to get std")
        loader_no_shuffle = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(chunks_train), batch_size=1, num_workers=args.num_workers, shuffle=False)
        loss, train_std = test_dropout(loader_no_shuffle)
        loader_no_shuffle = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(chunks_test), batch_size=1, num_workers=args.num_workers, shuffle=False)
        loss, test_std = test_dropout(loader_no_shuffle)

        print(chunks_train.shape, torch.Tensor(np.stack(train_std)).shape)
        print("finish testing, generating dataset and save") 
        trainset_with_std = torch.utils.data.TensorDataset(chunks_train, torch.Tensor(np.stack(train_std)))
        testset_with_std = torch.utils.data.TensorDataset(chunks_test, torch.Tensor(np.stack(test_std)))
        torch.save((trainset_with_std, testset_with_std), args.custom_data)


    # analysis and plot
    if args.analysis is not None:
        # perform cut
        idx_anomaly, idx_normal = utils.cut(test_loss, 0.005)
        # plot hist and detect positions
        # utils.plot_hist(test_loss, fig_path=args.fig_path)
        # utils.plot_detect(test_loss, data_test, idx_anomaly, sample_std, fig_path=args.fig_path)
        utils.plot_level(test_loss, data_test, idx_anomaly, sample_std, fig_path=args.fig_path)

