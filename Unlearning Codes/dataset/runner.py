import torch
from utils import AverageMeter, di_constraint, dp_constraint, eo_constraint
from fairlearn.metrics import (equalized_odds_difference, 
                               demographic_parity_difference,  
                               equalized_odds_difference)
from model import prepare_model
from dataset import prepare_data
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
class Runner():
    def __init__(self, args,):
        self.args = args
        self.model = prepare_model(args)
        self.model = torch.nn.DataParallel(self.model).to(self.args.device)
        self.biased_train_loader, self.biased_val_loader, self.balanced_test_loader, self.ft_loader = prepare_data(args, self.trial)

        self.ce_loss = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                         lr=self.args.lr, 
                                         weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=args.lr_decay)

        self.Acc = AverageMeter()
        self.Acc_test = AverageMeter()
        self.Loss = AverageMeter()
        self.best_acc = 0
        self.best_epoch = 0 
        self.save_best = False

    def train(self):
        self.best_acc = 0
        self.best_epoch = 0 
        for ep in tqdm(range(self.args.epochs)):
            self.Acc.reset()
            self.Loss.reset()
            self.model.train()
            pred_lst, y_lst, a_lst = [], [], []
            for i, (x, y, a) in enumerate(self.biased_train_loader):
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                pred = outputs.max(1).indices
                loss = self.ce_loss(outputs, y.type(torch.long))
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                accuracy = (outputs.max(1).indices == y).sum()/len(y)
                self.Acc.update(accuracy.item())
                self.Loss.update(loss.item())
                pred_lst.append(pred.cpu().detach())
                y_lst.append(y.cpu())
                a_lst.append(a.cpu())
            pred_lst = torch.cat(pred_lst)  
            y_lst = torch.cat(y_lst)
            a_lst = torch.cat(a_lst)
            EO = equalized_odds_difference(y_lst, pred_lst, sensitive_features=a_lst)
            DP = demographic_parity_difference(y_lst, pred_lst, sensitive_features=a_lst)
            self.writer.add_scalar('Loss_train', self.Loss.mean, ep)
            self.writer.add_scalar('Accuracy_train', self.Acc.mean, ep)
            self.writer.add_scalar('EO_train', EO, ep)
            self.writer.add_scalar('DP_train', DP, ep)
            self.writer.add_scalar('lr', self.scheduler.get_lr()[0], ep)
            print('[train] epoch: %d\tloss: %.3f\t acc : %.3f\tEO: %.3f\tDP:%.3f' %(
                ep, self.Loss.mean, self.Acc.mean, EO, DP))
            
            self.test(which_loader='val_loader', only_acc=True)
            if self.Acc_test.mean > self.best_acc:
                self.best_acc = self.Acc_test.mean
                best_epoch = ep
                current_patience = self.args.patience
                self.best_model = self.model.state_dict()
            else:
                current_patience -= 1
                if current_patience == 0:
                    print("Early stopping at epoch %d" % best_epoch)
                    torch.save(self.best_model, self.args.pretrain_path +f'./biased_model_{self.trial}.pt')
                    print(f"Saved best model to {self.args.pretrain_path}" +f'./biased_model_{self.trial}.pt')
                    self.save_best = True
                    break
            # print('[ val ] epoch: %d\tloss: %.3f\tacc : %.3f\tEO: %.3f\tDP:%.3f' %(
            #     ep, self.Loss.mean, self.Acc_test.mean, EO, DP))
        if not self.save_best:
            torch.save(self.model.state_dict(), self.args.pretrain_path +f'/biased_model_{self.trial}.pt')
            print("Saved checkpoint to %s" % self.args.pretrain_path +f'/biased_model_{self.trial}.pt')

    def test(self):
        self.Acc_test.reset()
        pred_lst, y_lst, a_lst, out_lst = [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for _, (x, y, a) in enumerate(self.balanced_test_loader):
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                outputs = self.model(x)
                pred = outputs.max(1).indices
                accuracy = (pred == y).sum()/len(y)
                self.Acc_test.update(accuracy.item())
                out_lst.append(outputs.cpu().detach())
                pred_lst.append(pred.cpu().detach())
                y_lst.append(y.cpu())
                a_lst.append(a.cpu())
            out_lst = torch.cat(out_lst)
            pred_lst = torch.cat(pred_lst)  
            y_lst = torch.cat(y_lst)
            a_lst = torch.cat(a_lst)

        print('[test] performance on Acc: %.3f' % ( self.Acc_test.mean))
        result = {'acc': self.Acc_test.mean}
        print(f'Accuracy_{self.Acc_test.mean}\t_Acc_0_{self.Acc_test_0.mean}\t_Acc_1_{self.Acc_test_1.mean}')


        # 这里开始计算fairness metric的值，常用的就是equalized_odds_difference, demographic_parity_difference
        for metric in [equalized_odds_difference, demographic_parity_difference]:
            value = metric(y_lst, pred_lst, sensitive_features=a_lst)
            print(f'{metric.__name__}:\t{value:.4f}')
            self.writer.add_scalar(f'{metric.__name__}', value)
            result[metric.__name__] = value
        for metric in [average_precision_score, roc_auc_score]:
            value = metric(y_lst, out_lst[:,1])
            print(f'{metric.__name__}:\t{value:.4f}')
            self.writer.add_scalar(f'{metric.__name__}', value)
            result[metric.__name__] = value
        return result