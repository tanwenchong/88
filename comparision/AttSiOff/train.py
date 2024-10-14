import os
#os.environ['CUDA_VISIBLE_dEVICES']='6,7'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='7'
import warnings
warnings.filterwarnings('ignore')
import torch
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
from torch import nn
import numpy as np
import pdb
import argparse
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, roc_auc_score,mean_squared_error, mean_absolute_error
from tqdm import tqdm
from load_datas import get_dataloader_rt_or_inter,get_dataloader_for_all_condition
from utils import *
modules_attn = __import__('model')

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=12)
parser.add_argument("--train_batch_size", type=int, default=128)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--sample_internal', type=int, default=10)
args = parser.parse_args()
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
     torch.use_deterministic_algorithms(True)
setup_seed(12)

class K_Train_Test():
    def __init__(self,args):
        super(K_Train_Test, self).__init__()
        self.args = args

    def model_train(self, train_loader, valid_loader, test_loader, patent_loader, testset_type):
        print(' --- testset type : {} ------'.format(testset_type))
        model = getattr(modules_attn, 'RNAFM_SIPRED_2')(dp=0.1, device=device).to(torch.float32).to(device)
        
        crepochion_mae = torch.nn.L1Loss(reduction='mean')
        crepochion_mse = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=4)

        model_name = 'model_' + testset_type.replace('_', '') + '.pth.tar'
        save_path = os.path.join('./output/', str(testset_type))
        os.makedirs(save_path, exist_ok=True)
        early_stopping = EarlyStopping(save_path, model_save_name=model_name, patience=20, greater=True)  # 早停的指标为pcc，所以得分越大越好

        
        for epoch in tqdm(range(self.args.n_epochs)):
            model.train()

            for id, inputs in enumerate(train_loader):
                for k in inputs.keys():
                    inputs[k] = inputs[k].to(device).to(torch.float32)

                pred = model(inputs)
                label = inputs['inhibit']
                loss = crepochion_mse(label, pred)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
                optimizer.step()

            scheduler.step()

            model.eval()
            with torch.no_grad():
                for id, inputs in enumerate(valid_loader):
                    for k in inputs.keys():
                        inputs[k] = inputs[k].to(device).to(torch.float32)

                    pred = model(inputs)
                    label = inputs['inhibit']
                    pred = np.array(pred.detach().cpu().numpy()).reshape(1, -1)[0]
                    label = np.array(label.detach().cpu().numpy()).reshape(1, -1)[0]
                    PCC, SPCC = round(stats.pearsonr(pred,label)[0], 3), round(stats.spearmanr(pred,label)[0], 3)

            early_stopping(SPCC, epoch, model)
            if early_stopping.early_stop:
                break


        model.load_state_dict(torch.load(early_stopping.best_model_path), strict=True)
        model.eval()

        with torch.no_grad():
            test_loaders = [valid_loader, test_loader, patent_loader]
            all_results = []
            for idx, test_loader in enumerate(test_loaders):
                all_preds = []
                all_labels = []
                for id, inputs in enumerate(test_loader):
                    for k in inputs.keys():
                        inputs[k] = inputs[k].to(device).to(torch.float32)

                    pred = model(inputs)
                    label = inputs['inhibit']
                    all_preds.extend(pred.detach().cpu().numpy().flatten())
                    all_labels.extend(label.detach().cpu().numpy().flatten())

                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)
                all_results.append(all_preds)
                PCC = round(stats.pearsonr(all_preds, all_labels)[0], 3)
                SPCC = round(stats.spearmanr(all_preds, all_labels)[0], 3)
                MSE = mean_squared_error(all_labels, all_preds)
                MAE = mean_absolute_error(all_labels, all_preds)
                AUC = roc_auc_score((all_labels > 0.5), all_preds)
                results = [f'{PCC:.4f}', f'{SPCC:.4f}', f'{AUC:.4f}', f'{MSE:.4f}', f'{MAE:.4f}']
                if idx == 0:
                    vPCCs, vSPCCs, vAUCs, vMSEs, vMAEs = results
                elif idx == 1:
                    tPCCs, tSPCCs, tAUCs, tMSEs, tMAEs = results
                elif idx == 2:
                    pPCCs, pSPCCs, pAUCs, pMSEs, pMAEs = results

        return vPCCs, vSPCCs, vAUCs, vMSEs, vMAEs, tPCCs, tSPCCs, tAUCs, tMSEs, tMAEs, pPCCs, pSPCCs, pAUCs, pMSEs, pMAEs,all_results           

            # plot_scatter_curve(pred,label,save_path,title='test_pcc_{}_spcc_{}'.format(PCC, SPCC))

    def all_condition(self, train_loader, valid_loader, test_loader, patent_loader):
        #train_loader, valid_loader, test_loader, patent_loader = get_dataloader_for_all_condition(self.args, train_csv, valid_csv, test_csv, patent_csv)
        vPCC, vSPCC, vAUC, vMSE, vMAE, tPCC, tSPCC, tAUC, tMSE, tMAE, pPCC, pSPCC, pAUC, pMSE, pMAE,all_results=self.model_train(train_loader, valid_loader, test_loader, patent_loader, 'all_condition')
        return vPCC, vSPCC, vAUC, vMSE, vMAE, tPCC, tSPCC, tAUC, tMSE, tMAE, pPCC, pSPCC, pAUC, pMSE, pMAE,all_results
def main(args):
    k_cross_course = K_Train_Test(args)
    test_csv = '/public2022/tanwenchong/rna/EnSIRNA-main/919/test.csv'
    patent_csv = '/public2022/tanwenchong/rna/EnSIRNA-main/919/patent2.csv'
    test_loader = get_dataloader_for_all_condition(args, test_csv, shuffle=False, drop_last=False)
    patent_loader = get_dataloader_for_all_condition(args, patent_csv, shuffle=False, drop_last=False)
    vPCCs, vSPCCs, vAUCs, vMSEs, vMAEs = [], [], [], [], []
    tPCCs, tSPCCs, tAUCs, tMSEs, tMAEs = [], [], [], [], []
    pPCCs, pSPCCs, pAUCs, pMSEs, pMAEs = [], [], [], [], []  
    df_test=pd.DataFrame()
    df_patent=pd.DataFrame()    
    for i in range(5):
        train_csv = '/public2022/tanwenchong/rna/EnSIRNA-main/919/train_'+str(i+1)+'.csv'
        valid_csv = '/public2022/tanwenchong/rna/EnSIRNA-main/919/valid_'+str(i+1)+'.csv'
        train_loader = get_dataloader_for_all_condition(args, train_csv)
        valid_loader = get_dataloader_for_all_condition(args, valid_csv)
        vPCC, vSPCC, vAUC, vMSE, vMAE, tPCC, tSPCC, tAUC, tMSE, tMAE, pPCC, pSPCC, pAUC, pMSE, pMAE,all_results=k_cross_course.all_condition(train_loader, valid_loader, test_loader, patent_loader)
        df_test[f'result_{i}']=all_results[1]
        df_patent[f'result_{i}']=all_results[2]
        print(len(all_results[1]),len(all_results[2]))

        vPCCs.append(vPCC)
        vSPCCs.append(vSPCC)
        vAUCs.append(vAUC)
        vMSEs.append(vMSE)
        vMAEs.append(vMAE)

        tPCCs.append(tPCC)
        tSPCCs.append(tSPCC)
        tAUCs.append(tAUC)
        tMSEs.append(tMSE)
        tMAEs.append(tMAE)

        pMSEs.append(pMSE)
        pMAEs.append(pMAE)
        pAUCs.append(pAUC)
        pPCCs.append(pPCC)
        pSPCCs.append(pSPCC)
    df_test.to_csv('/public2022/tanwenchong/rna/Attsioff/AttSiOff/paper_result/test.csv',index=False)
    df_patent.to_csv('/public2022/tanwenchong/rna/Attsioff/AttSiOff/paper_result/patent.csv',index=False)
    print(','.join(vMSEs),' ',f'MEAN: {np.array([float(i) for i in vMSEs]).mean():.4f}','MSE')
    print(','.join(vMAEs),' ',f'MEAN: {np.array([float(i) for i in vMAEs]).mean():.4f}','MAE')
    print(','.join(vAUCs),' ',f'MEAN: {np.array([float(i) for i in vAUCs]).mean():.4f}','AUC')
    print(','.join(vPCCs),' ',f'MEAN: {np.array([float(i) for i in vPCCs]).mean():.4f}','PCC')
    print(','.join(vSPCCs),' ',f'MEAN: {np.array([float(i) for i in vSPCCs]).mean():.4f}','SCC')

    print(','.join(tMSEs),' ',f'MEAN: {np.array([float(i) for i in tMSEs]).mean():.4f}','MSE')
    print(','.join(tMAEs),' ',f'MEAN: {np.array([float(i) for i in tMAEs]).mean():.4f}','MAE')
    print(','.join(tAUCs),' ',f'MEAN: {np.array([float(i) for i in tAUCs]).mean():.4f}','AUC')
    print(','.join(tPCCs),' ',f'MEAN: {np.array([float(i) for i in tPCCs]).mean():.4f}','PCC')
    print(','.join(tSPCCs),' ',f'MEAN: {np.array([float(i) for i in tSPCCs]).mean():.4f}','SCC')

    print(','.join(pMSEs),' ',f'MEAN: {np.array([float(i) for i in pMSEs]).mean():.4f}','MSE')
    print(','.join(pMAEs),' ',f'MEAN: {np.array([float(i) for i in pMAEs]).mean():.4f}','MAE')
    print(','.join(pAUCs),' ',f'MEAN: {np.array([float(i) for i in pAUCs]).mean():.4f}','AUC')
    print(','.join(pPCCs),' ',f'MEAN: {np.array([float(i) for i in pPCCs]).mean():.4f}','PCC')
    print(','.join(pSPCCs),' ',f'MEAN: {np.array([float(i) for i in pSPCCs]).mean():.4f}','SCC')  

    print('\n task finished!! \n')

if __name__ == '__main__':
    main(args)

