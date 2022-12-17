def train(model, train_dataloader, loss_fn, optimizer):
    model.train()
    for i, sample in enumerate(train_dataloader):
        x = sample['x'] # batched input image
        y = sample['y'] # batched ground truth

        pred = model(x) # model forward pass
        loss = loss_fn(pred, y) # converts to probability auto
        optimizer.zero_grad() # set inner-gradient relted param to 0
        loss.backward() # gradient (dw) back propagation
        optimizer.step() # Wt+1 <- Wt - lr*dw

# TODO:
# verbose = 1 loss printing
# learning graph visualise
# test function

def test():
