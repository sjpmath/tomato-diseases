def train(epoch, max_epoch, model, train_dataloader, loss_fn, optimizer, writer):
    model.train() # train mode
    running_loss = 0.0
    PRINT_EVERY=50

    pred_ls = []
    gt_ls = [] # list of tensors

    for i, sample in enumerate(train_dataloader):
        x = sample['x'] # batched input image
        y = sample['y'] # batched ground truth

        pred = model(x) # model forward pass
        loss = loss_fn(pred, y) # converts to probability auto
        optimizer.zero_grad() # set inner-gradient relted param to 0
        loss.backward() # gradient (dw) back propagation
        optimizer.step() # Wt+1 <- Wt - lr*dw

        running_loss += loss.item() # change to float from tensor

        pred_ls.append(pred)
        gt_ls.append(y)

        if i%PRINT_EVERY ==0 and i > 0: # every 50 iterations
            running_loss /= PRINT_EVERY
            print('[{}/{}]=[{}/{}] Loss: {:.4f}'.format(epoch, max_epoch, i, len(train_dataloader), running_loss))
            running_loss = 0.0

    # accuracy calc
    pred = torch.cat(pred_ls) # 100k x 3 x 224 x 224
    gt = torch.cat(gt_ls) # 100k x 1

    acc = torch.Tensor(pred.argmax(dim=0) == gt).mean().item() # [True, False,...]
    print('Train summary [{}/{}]: Accuracy: {.:2f}'.format(epoch, max_epoch, acc))

# TODO:
# verbose = 1 loss printing
# learning graph visualise
# test function

def test(epoch, max_epoch, model, valid_dataloader, writer):
    model.eval() # eval mode
    pred_ls = []
    gt_ls = [] # list of tensors

    for i, sample in enumerate(valid_dataloader):
        x = sample['x'] # batched input image
        y = sample['y'] # batched ground truth

        pred = model(x) # model forward pass

        pred_ls.append(pred)
        gt_ls.append(y)

    # accuracy calc
    pred = torch.cat(pred_ls) # 100k x 3 x 224 x 224
    gt = torch.cat(gt_ls) # 100k x 1

    acc = torch.Tensor(pred.argmax(dim=0) == gt).mean().item() # [True, False,...]
    print('Train summary [{}/{}]: Accuracy: {.:2f}'.format(epoch, max_epoch, acc))




    return acc









#
