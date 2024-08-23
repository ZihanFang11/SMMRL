

import torch.nn as nn
import warnings
import time
import random
import numpy as np
import torch
from tqdm import tqdm
from util.hypergraph_utils import construct_H_with_KNN_multi,generate_Lap_from_H
from util.loadMatData import load_data_semi
from SMMRL_model import CombineNet
import argparse
from util.utils import  get_evaluation_results
def train(args, device):

    ACC_mean_std=[]
    P_mean_std=[]
    R_mean_std=[]
    f1_micro_mean_std=[]
    f1_macro_mean_std=[]
    time_list = []
    LX={}
    for j in range(args.n_view):
        LX[j] = torch.norm( feature_train[j].t().matmul(feature_train[j]))


    for kk in range(args.resp):
        model = CombineNet(args.n_feats, args.n_view,args.n_classes, LX,device,args).to(device)
        criterion=nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        loss_list = []
        begin_time = time.time()
        with tqdm(total=args.epoch, desc="Training") as pbar:
            for epoch in range(1,args.epoch+1):
                model.train()
                output = model(feature_train,lap_train)
                loss=criterion(output, labels_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

                if epoch%20==0:
                    print(f'Epoch:{epoch}\tLoss:{loss.item()}')
                pbar.update(1)

        cost_time = time.time() - begin_time
        time_list.append(cost_time)
        with torch.no_grad():
            model.eval()
            output=model.infer(feature_test)
        pred_labels = torch.argmax(output, 1).cpu().detach().numpy()

        ACC, P, R, F1_macro, F1_micro = get_evaluation_results(labels_test.cpu().detach().numpy(),
                                                               pred_labels)
        ACC_mean_std.append(ACC)
        f1_macro_mean_std.append(F1_macro)
        f1_micro_mean_std.append(F1_micro)
        P_mean_std.append(P)
        R_mean_std.append(R)
        time_list.append(cost_time)
        print("------------------------")
        print("ACC:   {:.2f}".format(ACC * 100))
        print("F1_macro:   {:.2f}".format(F1_macro * 100))
        print("F1_micro:   {:.2f}".format(F1_micro * 100))
        print("------------------------")


    acc = str(round(np.mean(ACC_mean_std) * 100, 2)) + "(" + str(round(np.std(ACC_mean_std) * 100, 2)) + ")"
    P = str(round(np.mean(P_mean_std) * 100, 2)) + "(" + str(round(np.std(P_mean_std) * 100, 2)) + ")"
    R = str(round(np.mean(R_mean_std) * 100, 2)) + "(" + str(round(np.std(R_mean_std) * 100, 2)) + ")"

    f1_macro = str(round(np.mean(f1_macro_mean_std) * 100, 2)) + "(" + str(
        round(np.std(f1_macro_mean_std) * 100, 2)) + ")"
    f1_micro = str(round(np.mean(f1_micro_mean_std) * 100, 2)) + "(" + str(
        round(np.std(f1_micro_mean_std) * 100, 2)) + ")"

    Runtime_mean_std = str(round(np.mean(time_list), 2)) + "(" + str(
        round(np.std(time_list), 2)) + ")"

    if args.save_total_results:
        with open(args.save_file, "a") as f:

            f.write("{}:{}\n".format(args.data, dict(
                zip(['acc', 'F1_macro', 'F1_micro', 'precision', 'recall', 'time', ],
                    [acc, f1_macro, f1_micro, P, R, Runtime_mean_std]))))

def parameter_parser():
    """
    Parses the arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./data/", help="Path of datasets.")
    parser.add_argument("--data", type=str, default="BDGP", help="Name of datasets.")
    parser.add_argument("--save_file", type=str, default="result.txt")

    parser.add_argument("--fix_seed", action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=40, help="Random seed, default is 42.")
    parser.add_argument("--device", default="0", type=str, required=False)

    parser.add_argument("--norm", action='store_true', default=True, help="Normalize the feature.")
    parser.add_argument("--train_ratio", type=float, default=0.1, help="Number of labeled samples per classes")
    parser.add_argument("--resp", type=int, default=10, help="Number of labeled samples per classes")

    parser.add_argument("--save_total_results", action='store_true', default=True, help="Save experimental result.")

    parser.add_argument('--epoch', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--block', type=int, default=1, help='block')
    parser.add_argument('--knn', type=int, default=10, help='block')
    parser.add_argument('--theta1', type=float, default=0.01)
    parser.add_argument('--theta2', type=float, default=0.1)
    parser.add_argument('--lamb', type=float, default=1, help='lambda')
    parser.add_argument("--fusion_type", type=str, default="weight")  # trust,average,weight,attention
    parser.add_argument("--batch_size", default=200, help='number of validate dataset', type=int, required=False)
    parser.add_argument("--num_test", default=1000, help='number of test dataset', type=int, required=False)

    return parser.parse_args()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parameter_parser()
    seed = args.seed
    path = args.path
    args.device = '0'
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)

    dataset_dict = {1: 'BDGP', 2: 'esp_game', 3: 'flickr', 4: 'HW', 5: 'NUSWide20k', 6: 'smallReuters'}
    select_dataset = [1,2,3,4,5,6]

    for i in select_dataset:
            args.data = dataset_dict[i]
            print("========================",args.data)


            feature_list, labels, idx_train, idx_test = load_data_semi(args, args.data, device)

            print(len(idx_train), len(idx_test))
            n_view = len(feature_list)
            n_feats = [x.shape[1] for x in feature_list]
            n = feature_list[0].shape[0]
            n_classes = len(np.unique(labels))
            args.n_view=n_view
            args.n_feats=n_feats
            args.n_classes=n_classes


            print(args.data, n, n_view, n_feats)
            feature_train = []
            feature_test = []
            for i in range(n_view):
                feature_train.append(feature_list[i][idx_train] / 1.0)
                feature_test.append(feature_list[i][idx_test] / 1.0)

            labels_test = labels[idx_test].to(device)
            labels_train= labels[idx_train].to(device)


            H_train = construct_H_with_KNN_multi(feature_train, args.knn, True)
            lap_train = generate_Lap_from_H(H_train)

            for i in range(n_view):
                feature_train[i] = torch.from_numpy(feature_train[i]).float().to(device)
                feature_test[i] = torch.from_numpy(feature_test[i]).float().to(device)
                lap_train[i] = torch.from_numpy(lap_train[i]).float().to(device)



            if args.fix_seed:
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed(args.seed)

            res = train(args, device)

