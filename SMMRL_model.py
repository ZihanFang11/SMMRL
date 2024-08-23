import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn as nn


class CombineNet(nn.Module):
    def __init__(self,nfeats, n_view,n_class, LX,device,args):
        super(CombineNet, self).__init__()

        self.n_class =n_class
        self.blocks = args.block
        self.n_view=n_view
        self.lamb=args.lamb
        self.device=device
        self.theta1 = nn.Parameter(torch.FloatTensor([args.theta1]), requires_grad=True).to(device)
        self.theta2 = nn.Parameter(torch.FloatTensor([args.theta2]), requires_grad=True).to(device)
        self.ZZ_init = nn.ModuleList([nn.Linear(feat,  self.n_class).to(device) for feat in nfeats])
        self.U = nn.ModuleList([nn.Linear(self.n_class, self.n_class).to(device) for _ in range(self.n_view)])
        self.fusionlayer = FusionLayer(n_view, args.fusion_type, self.n_class, hidden_size=64)
        self.LX=LX
        self.M = {}
    def  self_active_l1(self, u, theta):
        return F.selu(u - theta) - F.selu(-1.0 * u - theta)
    def self_active_l21(self, x, theta):
        nw = torch.norm(x)
        if nw > theta:
            x = (nw -theta) * x / nw
        else:
            x = torch.zeros_like(x)
        return x
    def forward(self, features, lap):
        Z = list()
        out_tmp  = torch.zeros(size=[features[0].size(0), self.n_class]).to(self.device)
        for j in range(self.n_view):
            out_tmp += self.ZZ_init[j](features[j] / 1.0)
        Z.append(self.self_active_l1(out_tmp / self.n_view, self.theta1))
        lapx = list()

        for j in range(self.n_view):
            lapx.append(torch.mm(torch.mm(features[j].T,lap[j]),features[j]))
            self.M[j]=self.self_active_l21(torch.mm(features[j].T,Z[-1])/self.LX[j],self.theta2)

        for i in range( self.blocks):
            Z_tmp =list()
            for j in range(self.n_view):

                input1=self.U[j](self.M[j])
                input2=torch.mm(lapx[j],self.M[j])/self.LX[j]*self.lamb
                input3=torch.mm(features[j].T,Z[-1])/self.LX[j]
                M_tmp=input1-input2+input3

                self.M[j] =self.self_active_l21(M_tmp, self.theta2)
                Z_tmp.append(self.self_active_l1(torch.mm(features[j],self.M[j]), self.theta1))
            z = self.fusionlayer(Z_tmp)
            Z.append(z)
        out=Z[-1]
        return out

    def infer(self, features):
        Z = list()
        for i in range(self.blocks):
            Z_tmp = list()
            for j in range(self.n_view):
                Z_tmp.append(self.self_active_l1(torch.mm(features[j],self.M[j]), self.theta1))
            z = self.fusionlayer(Z_tmp)
            Z.append(z)
        out = Z[-1]
        return out


class FusionLayer(nn.Module):
    def __init__(self, num_views, fusion_type, in_size, hidden_size=64):
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        if self.fusion_type == 'weight':
            self.weight = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)
        if self.fusion_type == 'attention':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 32, bias=False),
                nn.Tanh(),
                nn.Linear(32, 1, bias=False)
            )

    def forward(self, emb_list):
        if self.fusion_type == "average":
            common_emb = sum(emb_list) / len(emb_list)
        elif self.fusion_type == "weight":
            weight = F.softmax(self.weight, dim=0)
            common_emb = sum([w * e for e, w in zip(weight, emb_list)])
        elif self.fusion_type == 'attention':
            emb_ = torch.stack(emb_list, dim=1)
            w = self.encoder(emb_)
            weight = torch.softmax(w, dim=1)
            common_emb = (weight * emb_).sum(1)
        else:
            sys.exit("Please using a correct fusion type")
        return common_emb
