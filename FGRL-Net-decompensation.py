import numpy as np
import argparse
import os
import pickle
from datetime import datetime
import random
from os.path import exists

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True

from utils import utils
from utils.readers import DecompensationReader
from utils.preprocessing_level import Discretizer, Normalizer
from utils import metrics
from utils import common_utils


class FGRLNet(nn.Module):
    def __init__(self, input_dim, row_dim, hidden_dim, output_dim, task='Decompensation',):
        super(FGRLNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.row_dim = row_dim
        self.task = task
        # self.keep_prob = keep_prob

        self.output0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output1 = nn.Linear(self.hidden_dim, self.output_dim)

        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()
        
        self.gru=nn.GRU(input_dim , self.hidden_dim , batch_first=True)
        
        self.dowmsample1D=nn.MaxPool1d(2,stride=2) 
        self.conv1D1 = nn.Conv1d(input_dim, input_dim, kernel_size=3,padding = 1)#
        self.conv1D2 = nn.Conv1d(input_dim, input_dim, kernel_size=5,padding = 2)#
        # self.conv1D3 = nn.Conv1d(input_dim, input_dim, kernel_size=3,padding = 1)#

        self.conv1D_feature1 = nn.Conv1d(row_dim, row_dim, kernel_size=3,padding = 1)#
        self.conv1D_feature2 = nn.Conv1d(row_dim, row_dim, kernel_size=5,padding = 2)#
        # self.conv1D_feature3 = nn.Conv1d(row_dim, row_dim, kernel_size=3,padding = 1)#

    def forward(self, batch_feature):
        batch_size = batch_feature.size(0)
        row = batch_feature.size(1)
        feature_dim = batch_feature.size(2)
        # input shape [batch_size, timestep, feature_dim] 
        
        # conv time dimention
        # print('shape of batch_feature',batch_feature.shape)
        time_conv = batch_feature.permute(0, 2, 1)
        # print('shape of conv_input',conv_input.shape)

        time_conv = self.conv1D1(time_conv)
        time_conv = self.conv1D2(time_conv)

        time_conv = time_conv.permute(0, 2, 1)
        time_conv = torch.mul(batch_feature, self.sigmoid(time_conv))       
        
        # conv feature dimention
        row_diff = self.row_dim - row
        if row_diff > 0:
            # print('shape of in_feature',in_feature.shape)            
            pad = nn.ZeroPad2d(padding=(0, 0, 0, row_diff))
            feature_conv = pad(batch_feature)
            # print('after pad',in_feature.shape)

            feature_conv = self.conv1D_feature1(feature_conv)
            feature_conv = self.conv1D_feature2(feature_conv)
            feature_conv = feature_conv[:,:row,:]
            # print('feature_conv ',feature_conv.shape)
        else:
            feature_conv = self.conv1D_feature1(batch_feature)  
            feature_conv = self.conv1D_feature2(feature_conv) 
    
        feature_conv = torch.mul(batch_feature, self.sigmoid(feature_conv))

        
        in_feature = torch.add(time_conv,feature_conv)

        output, h = self.gru(in_feature)
        if self.task == 'Decompensation':
            output = torch.squeeze(output, dim=0)
        else:
            output = torch.squeeze(h, dim=0)

        output = self.output1(self.relu(self.output0(output)))# b 1
        output = self.sigmoid(output)
         
        return output


def parse_arguments(parser):
    parser.add_argument('--test_mode', type=int, default=0, help='Test SA-CRNN on MIMIC-III dataset')
    # parser.add_argument('--data_path', type=str, default='data/decompensation/', help='The path to the MIMIC-III data directory')
    # parser.add_argument('--file_name', type=str, default='AdaCare', help='File name to save model')
    parser.add_argument('--cuda', type=str, default='cuda:0', help='cuda')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learing rate')

    parser.add_argument('--input_dim', type=int, default=76, help='Dimension of visit record data')
    parser.add_argument('--rnn_dim', type=int, default=384, help='Dimension of hidden units in RNN')
    parser.add_argument('--output_dim', type=int, default=1, help='Dimension of prediction target')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--r_visit', type=int, default=4, help='Compress ration r for visit features')
    parser.add_argument('--r_conv', type=int, default=4, help='Compress ration r for convolutional features')
    parser.add_argument('--kernel_size', type=int, default=2, help='Convolutional kernel size')
    parser.add_argument('--kernel_num', type=int, default=64, help='Number of convolutional filters')
    parser.add_argument('--activation_func', type=str, default='sigmoid', help='Activation function for feature recalibration (sigmoid / sparsemax)')

    args = parser.parse_args()
    return args

def get_loss(y_pred, y_true, mask):
    # loss_fun = torch.nn.BCELoss()
    # loss = loss_fun(y_pred, y_true)
    masked_y_pred = y_pred * mask

    loss = y_true * torch.log(masked_y_pred + 1e-7) + (1 - y_true) * torch.log(1 - masked_y_pred + 1e-7)
    loss = torch.sum(loss, dim=1) / torch.sum(mask, dim=1)
    loss = torch.neg(torch.sum(loss))

    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    hidden_dim = 384
    row_dim = 400

    demo_dim = 12
    diagnoses_dim = 128
    feature_dim =96
    input_dim = demo_dim + diagnoses_dim + feature_dim

    file_model = './saved_weights/'+"knowledgeCare-decom-layerT2F2-c35.model"

    data_path = 'data/decompensation/'
    file_discretizer_header = data_path + 'discretizer_header_level.pk'
    file_normalizer_state = data_path + 'decomp_ts_1.00_impute_previous_start_zero_masks_True_n_2908075_level.normalizer'
    file_train_data_raw = data_path + 'train_data_raw_level.pk'
    file_val_data_raw = data_path + 'val_data_raw_level.pk'
    file_test_data_raw = data_path + 'test_data_raw_level.pk'

    # load data for train and test process
    demographic_data = pickle.load(open(data_path + 'demographic_data.pk', 'rb'))
    diagnosis_data = pickle.load(open(data_path + 'diagnosis_data.pk', 'rb'))
    idx_list = pickle.load(open(data_path + 'idx_list.pk', 'rb'))


    # debug = True
    debug = False

    if args.test_mode == 1:
        print('Preparing test data ... ')

        discretizer = Discretizer(timestep=1.0, store_masks=True,
                                impute_strategy='previous', start_time='zero')

        # discretizer_header = discretizer.transform(train_data_loader._data["X"][0])[1].split(',')
        discretizer_header = pickle.load(open(file_discretizer_header, 'rb'))        
        print('load discretizer_header success!')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

        normalizer = Normalizer(fields=cont_channels)
        normalizer.load_params(file_normalizer_state)

        if exists(file_test_data_raw):
            test_data_raw = pickle.load(open(file_test_data_raw, 'rb'))               
            print('load test_data_raw success! test count =',len(test_data_raw['data'][1]))
        else:
            test_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(data_path, 'test'),
                                                                    listfile=os.path.join(data_path, 'test_listfile.csv'), small_part=args.small_part)            
            test_data_raw = utils.get_data_raw(test_data_loader, discretizer, normalizer)
            pickle.dump(test_data_raw, open(file_test_data_raw, 'wb'), -1)
     
        test_data_gen = utils.BatchGenDeepSupervision(test_data_raw, args.batch_size, shuffle=False, return_names=True)

        print('Constructing model ... ')
        device = torch.device(args.cuda if torch.cuda.is_available() == True else 'cpu')
        print("available device: {}".format(device))

        model = FGRLNet(input_dim, row_dim, hidden_dim, output_dim = 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        checkpoint = torch.load(file_model)
        # checkpoint = torch.load(file_name,map_location=torch.device('cpu'))
        
        save_chunk = checkpoint['chunk']
        print("last saved model is in chunk {}".format(save_chunk))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.eval()
        with torch.no_grad():
            cur_test_loss = []
            test_true = []
            test_pred = []
            
            for each_batch in range(test_data_gen.steps):
                batch_data = next(test_data_gen)
                batch_name = batch_data['names']
                batch_data = batch_data['data']

                batch_x = torch.tensor(batch_data[0][0], dtype=torch.float32).to(device)
                batch_mask = torch.tensor(batch_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
                batch_y = torch.tensor(batch_data[1], dtype=torch.float32).to(device)
                
                if batch_mask.size()[1] > 400:
                    batch_x = batch_x[:, :400, :]
                    batch_mask = batch_mask[:, :400, :]
                    batch_y = batch_y[:, :400, :]
                
                batch_size =  batch_x.size()[0]
                test_time = batch_mask.size()[1]
            
                batch_demo = []
                batch_diagnosis = []    # b * n_medical_codes => 256 * 128
                for i in range(len(batch_name)):
                    cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                    cur_idx = cur_id + '_' + cur_ep
                    cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
                    batch_demo.append(cur_demo)

                    # Add diagnosis
                    cur_diagnosis = torch.tensor(diagnosis_data[idx_list.index(cur_idx)], dtype=torch.float32)
                    batch_diagnosis.append(cur_diagnosis)
                
                batch_demo = torch.stack(batch_demo).to(device)# b * 12 => 128 * 12
                # print('batch_demo shape = ',batch_demo.shape)
                batch_diagnosis = torch.stack(batch_diagnosis).to(device)# b * 12 => 128 * 128
                # print('batch_diagnosis shape = ',batch_diagnosis.shape)
                batch_demo = torch.cat((batch_demo,batch_diagnosis),1)                
                batch_demo = batch_demo.repeat(1,test_time)
                batch_demo = batch_demo.reshape(batch_size, test_time, demo_dim + diagnoses_dim)

                # optimizer.zero_grad()
                in_feature = torch.cat((batch_demo,batch_x),2) 
                output = model(in_feature)

                loss = get_loss(output, batch_y, batch_mask)
                cur_test_loss.append(loss.cpu().detach().numpy()) 
                
                for m, t, p in zip(batch_mask.cpu().numpy().flatten(), batch_y.cpu().numpy().flatten(), output.cpu().detach().numpy().flatten()):
                    if np.equal(m, 1):
                        test_true.append(t)
                        test_pred.append(p)
            
            print('Test loss = %.4f'%(np.mean(np.array(cur_test_loss))))
            print('\n')
            test_pred = np.array(test_pred)
            test_pred = np.stack([1 - test_pred, test_pred], axis=1)
            test_ret = metrics.print_metrics_binary(test_true, test_pred)

    else:
        ''' Prepare training data'''        
        print('Preparing training data ... ')


        if exists(file_train_data_raw):
            train_data_raw = pickle.load(open(file_train_data_raw, 'rb'))        
            print('load train_data_raw success! train count =',len(train_data_raw['data'][1]))

            discretizer_header = pickle.load(open(file_discretizer_header, 'rb'))        
            print('load discretizer_header success!')
            cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
        else:
            train_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(
                                data_path, 'train'), listfile=os.path.join(data_path, 'train_listfile.csv'), small_part=args.small_part)

            discretizer = Discretizer(timestep=1.0, store_masks=True, impute_strategy='previous', start_time='zero')

            discretizer_header = discretizer.transform(train_data_loader._data["X"][0])[1].split(',')
            pickle.dump(discretizer_header, open(file_discretizer_header, 'wb'), -1)
            print('Save discretizer_header success!')

            cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
            normalizer = Normalizer(fields=cont_channels)
            normalizer.load_params(file_normalizer_state)

            train_data_raw = utils.get_data_raw(train_data_loader, discretizer, normalizer)
            pickle.dump(train_data_raw, open(file_train_data_raw, 'wb'), -1)        

        train_data_gen = utils.BatchGenDeepSupervision(train_data_raw, args.batch_size, shuffle=True, return_names=True)

        if exists(file_val_data_raw):
            val_data_raw = pickle.load(open(file_val_data_raw, 'rb'))        
            print('load val_data_raw success! val count =',len(val_data_raw['data'][1]))
        else:
            val_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(
                                data_path, 'train'), listfile=os.path.join(data_path, 'val_listfile.csv'), small_part=args.small_part)

            val_data_raw = utils.get_data_raw(val_data_loader, discretizer, normalizer)
            pickle.dump(val_data_raw, open(file_val_data_raw, 'wb'), -1)  

        val_data_gen = utils.BatchGenDeepSupervision(val_data_raw, args.batch_size, shuffle=False, return_names=True)


        '''Model structure'''
        print('Constructing model ... ')
        device = torch.device(args.cuda if torch.cuda.is_available() == True else 'cpu')
        print("available device: {}".format(device))

        model = FGRLNet(input_dim, row_dim, hidden_dim, output_dim = 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        '''Train phase'''
        print('Start training ... ')

        train_loss = []
        val_loss = []
        batch_loss = []
        max_auprc = 0

        for each_chunk in range(args.epochs):
            cur_batch_loss = []
            model.train()
            for each_batch in range(train_data_gen.steps):
                batch_data = next(train_data_gen)
                batch_name = batch_data['names']
                batch_data = batch_data['data']

                if debug:
                    for i in range(len(batch_name)):
                        print('name = ',batch_name[i])

                
                batch_x = torch.tensor(batch_data[0][0], dtype=torch.float32).to(device)
                batch_mask = torch.tensor(batch_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
                batch_y = torch.tensor(batch_data[1], dtype=torch.float32).to(device)

              
                if batch_mask.size()[1] > 400:
                    batch_x = batch_x[:, :400, :]
                    batch_mask = batch_mask[:, :400, :]
                    batch_y = batch_y[:, :400, :]
                
                batch_size =  batch_x.size()[0]
                test_time = batch_mask.size()[1]
                
                
                batch_demo = []

                # Add diagnosis
                batch_diagnosis = []    # b * n_medical_codes => 128
                for i in range(len(batch_name)):
                    cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                    cur_idx = cur_id + '_' + cur_ep
                    cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
                    batch_demo.append(cur_demo)

                    # Add diagnosis
                    cur_diagnosis = torch.tensor(diagnosis_data[idx_list.index(cur_idx)], dtype=torch.float32)
                    batch_diagnosis.append(cur_diagnosis)
                
                batch_demo = torch.stack(batch_demo).to(device)# b * 12 => 128 * 12
                # print('batch_demo shape = ',batch_demo.shape)
                batch_diagnosis = torch.stack(batch_diagnosis).to(device)# b * 12 => 128 * 128
                # print('batch_diagnosis shape = ',batch_diagnosis.shape)
                batch_demo = torch.cat((batch_demo,batch_diagnosis),1)                
                batch_demo = batch_demo.repeat(1,test_time)
                # print('train batch_size = ',batch_size)
                batch_demo = batch_demo.reshape(batch_size, test_time, demo_dim + diagnoses_dim)

                optimizer.zero_grad()
                in_feature = torch.cat((batch_demo,batch_x),2) 
                output = model(in_feature)

                # print('output shape = ',output.shape)
                # print('batch_y shape = ',batch_y.shape)
                loss = get_loss(output, batch_y, batch_mask)
                cur_batch_loss.append(loss.cpu().detach().numpy())

                loss.backward()
                optimizer.step()
                
                if each_batch % 50 == 0:
                    print('Chunk %d, Batch %d: Loss = %.4f'%(each_chunk, each_batch, cur_batch_loss[-1]))

            batch_loss.append(cur_batch_loss)
            train_loss.append(np.mean(np.array(cur_batch_loss)))
            
            print("\n==>Predicting on validation")
            with torch.no_grad():
                model.eval()
                cur_val_loss = []
                valid_true = []
                valid_pred = []
                for each_batch in range(val_data_gen.steps):
                    batch_data = next(val_data_gen)
                    batch_name = batch_data['names']
                    batch_data = batch_data['data']
                    
                    batch_x = torch.tensor(batch_data[0][0], dtype=torch.float32).to(device)
                    batch_mask = torch.tensor(batch_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
                    batch_y = torch.tensor(batch_data[1], dtype=torch.float32).to(device)
                    
                    if batch_mask.size()[1] > 400:
                        batch_x = batch_x[:, :400, :]
                        batch_mask = batch_mask[:, :400, :]
                        batch_y = batch_y[:, :400, :]
                
                    batch_size =  batch_x.size()[0]
                    test_time = batch_mask.size()[1]
                    
                    batch_demo = []
                    batch_diagnosis = []    # b * n_medical_codes => 256 * 128
                    for i in range(len(batch_name)):
                        cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                        cur_idx = cur_id + '_' + cur_ep
                        cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
                        batch_demo.append(cur_demo)

                        # Add diagnosis
                        cur_diagnosis = torch.tensor(diagnosis_data[idx_list.index(cur_idx)], dtype=torch.float32)
                        batch_diagnosis.append(cur_diagnosis)
                    
                    batch_demo = torch.stack(batch_demo).to(device)# b * 12 => 128 * 12
                    batch_diagnosis = torch.stack(batch_diagnosis).to(device)# b * 12 => 128 * 128
                    batch_demo = torch.cat((batch_demo,batch_diagnosis),1)                
                    batch_demo = batch_demo.repeat(1,test_time)
                    
                    # print('val batch_size = ',batch_size)
                    batch_demo = batch_demo.reshape(batch_size, test_time, demo_dim + diagnoses_dim)

                    # optimizer.zero_grad()                    
                    in_feature = torch.cat((batch_demo,batch_x),2) 
                    output = model(in_feature)

                    loss = get_loss(output, batch_y, batch_mask)

                    cur_val_loss.append(loss.cpu().detach().numpy())

                    for m, t, p in zip(batch_mask.cpu().numpy().flatten(), batch_y.cpu().numpy().flatten(), output.cpu().detach().numpy().flatten()):
                        if np.equal(m, 1):
                            valid_true.append(t)
                            valid_pred.append(p)

                val_loss.append(np.mean(np.array(cur_val_loss)))
                print('Valid loss = %.4f'%(val_loss[-1]))
                print('\n')
                valid_pred = np.array(valid_pred)
                valid_pred = np.stack([1 - valid_pred, valid_pred], axis=1)
                ret = metrics.print_metrics_binary(valid_true, valid_pred)
                print()

                cur_auprc = ret['auprc']
                if cur_auprc > max_auprc:
                    max_auprc = cur_auprc
                    state = {
                        'net': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'chunk': each_chunk
                    }
                    torch.save(state, file_model)
                    print('\n------------ Save best model ------------\n')
