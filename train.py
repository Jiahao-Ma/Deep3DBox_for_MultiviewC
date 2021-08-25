from data.MultiviewCdataset import MultiviewC_dataset
from utils.config import opt
from torch.utils import data
from nets.model import Deep3DBox, OrientationLoss
import torch
from tqdm import tqdm

def main():
    dataset = MultiviewC_dataset( root = opt.root, split = opt.split, bins = opt.bins, mode='train')
    dataloader = data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    test_dataset = MultiviewC_dataset( root = opt.root, split = opt.split, bins = opt.bins, mode='test')
    test_dataloader = data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    
    model = Deep3DBox(opt.bins).cuda()
    opt_SGD = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
    conf_loss_func = torch.nn.CrossEntropyLoss().cuda()
    dim_loss_func = torch.nn.MSELoss().cuda()
    orient_loss_func = OrientationLoss

    start_epoch = 0
    if opt.load_model:
        checkpoint = torch.load(opt.model_path)
        model.load_state_dict(checkpoint['model'])
        opt_SGD.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        print('Found previous checkpoints from %s'%opt.model_path)
        print('Resuming training ...')
    
    for epoch in range(start_epoch, opt.epochs):

        with tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1} / {opt.epochs}', postfix=dict, mininterval=1) as pbar:
            model.train()
            loss_all = 0
            dim_loss_all = 0
            conf_loss_all = 0
            orient_loss_all = 0
            for i, (batch_img, batch_label) in enumerate(dataloader):
                true_orient = batch_label['Orientation'].float().cuda()
                true_conf = batch_label['Confidence'].long().cuda()
                true_dim = batch_label['Dimension'].float().cuda()

                batch_img = batch_img.float().cuda()
                [orient, conf, dim] = model(batch_img)

                orient_loss = orient_loss_func(orient, true_orient, true_conf)
                dim_loss = dim_loss_func(dim, true_dim)

                true_conf = torch.argmax(true_conf, dim=-1)
                conf_loss = conf_loss_func(conf, true_conf)

                loss_theta = conf_loss + opt.W * orient_loss 
                loss = opt.alpha * dim_loss  + loss_theta

                opt_SGD.zero_grad()
                loss.backward()
                opt_SGD.step()
                
                loss_all += loss.item()
                dim_loss_all += dim_loss.item()
                conf_loss_all += conf_loss.item()
                orient_loss_all += orient_loss.item()

                pbar.set_postfix(**{
                    'total_loss' :  '{:.4f}'.format(loss_all / (i + 1)),
                    'dim_loss' : '{:.4f}'.format(dim_loss_all / (i + 1)),
                    'conf_loss' : '{:.4f}'.format(conf_loss_all / (i + 1)),
                    'orient_loss' : '{:.4f}'.format(orient_loss_all / (i + 1)),
                })
                pbar.update(1)

        print('Start Validation.')
        with tqdm(total = len(test_dataloader), desc=f'Epoch {epoch + 1} / {opt.epochs}', postfix=dict, mininterval=0.4) as pbar:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for i, (batch_img, batch_label) in enumerate(test_dataloader):
                    true_orient = batch_label['Orientation'].float().cuda()
                    true_conf = batch_label['Confidence'].long().cuda()
                    true_dim = batch_label['Dimension'].float().cuda()

                    batch_img = batch_img.float().cuda()
                    [orient, conf, dim] = model(batch_img)

                    orient_loss = orient_loss_func(orient, true_orient, true_conf)
                    dim_loss = dim_loss_func(dim, true_dim)

                    true_conf = torch.argmax(true_conf, dim=-1)
                    conf_loss = conf_loss_func(conf, true_conf)

                    loss_theta = conf_loss + opt.W * orient_loss 
                    loss = opt.alpha * dim_loss  + loss_theta
                    
                    val_loss += loss.item()
                    pbar.set_postfix(**{
                        'total_loss' : '{:.4f}'.format(val_loss / (i + 1))
                    })
                    pbar.update(1)
        val_loss /= len(test_dataloader) 
        loss_all /= len(dataloader)
        print('Finish Validation.')
        print('Epoch: ', str(epoch + 1), '\tval_loss: ', val_loss)

        
        
        name = '.\weights\epoch{:02d}_train_loss{:.2f}_val_loss{:.2f}.pkl'.format(epoch, loss_all, val_loss)
        print("====================")
        print ("Done with epoch %s!" % epoch)
        print ("Saving weights as %s ..." % name)
        torch.save({
            'epoch' : epoch,
            'model' : model.state_dict(),
            'optimizer' : opt_SGD.state_dict(),
        }, name)
        print("====================")

if __name__ == '__main__':
    main()