import dgl
import dgl.function as dglfn
import dgl.nn.pytorch as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):

    def __init__(self):
        super().__init__()
        self.a_init = nn.Linear(82, 120)
        self.b_init = nn.Linear(12, 8)
        self.r_init_1 = nn.Linear(20, 9)
        self.r_init_2 = nn.Linear(9 + 21, 120)
        # message transport
        self.r_mt = MT(120)
        self.a_mt = MT(120)
        # ligand
        self.A = nn.Linear(120, 120)
        self.B = nn.Linear(120*2,120)
        self.a_conv1 = SConv1(120, 8, 120, 4)    
        self.a_conv2 = SConv1(120, 8, 120, 4)    
        # protein
        self.C = nn.Linear(120, 120)
        self.D = nn.Linear(120*2,120)
        self.r_conv1 = SConvr1(120, 0, 120, 4)    
        self.r_conv2 = SConvr1(120, 0, 120, 4)   
        # interaction
        self.i_conf = DistanceConv(120, 8, 3)
        # predict
        self.classifier = nn.Sequential(
            nn.Linear(120 + 120 + 120 + 6, 298),
            nn.PReLU(),
            nn.Linear(298, 160),
            nn.PReLU(),
            nn.Linear(160, 1)
        )

        self.sum_pool = dglnn.SumPooling()
        self.mean_pool = dglnn.AvgPooling()

    def forward(self, ga, gr, gi, vina): 
        device = torch.device("cuda:0")
        ga = ga.to('cuda:0')
        gr = gr.to('cuda:0')
        gi = gi.to('cuda:0')
        vina = vina.to('cuda:0')
          
        va_init = self.a_init(ga.ndata['feat'])
        ea = self.b_init(ga.edata['feat'])   
        vr = self.r_init_1(gr.ndata['feat'][:, :20])    
        vr = torch.cat((vr, gr.ndata['feat'][:, 20:]), -1)
        vr_init = self.r_init_2(vr) 

        vi_a = self.a_init(gi.ndata['feat']['atom'])   
        vi_r = self.r_init_1(gi.ndata['feat']['residue'][:, :20])  
        vi_r = torch.cat((vi_r, gi.ndata['feat']['residue'][:, 20:]), -1)
        vi_r = self.r_init_2(vi_r)   
        vi_init = torch.cat((vi_a, vi_r), dim=0)
        ei = gi.edata['weight'].reshape(-1)
        ei = torch.cat((ei, ei)).unsqueeze(1)  

        gii = dgl.add_reverse_edges(dgl.to_homogeneous(gi))  
        gii.set_batch_num_nodes(gi.batch_num_nodes('atom') + gi.batch_num_nodes('residue'))  
        gii.set_batch_num_edges(gi.batch_num_edges() * 2)  
        va = self.a_mt(gr, vr_init, ga, va_init)         
        vr = self.r_mt(ga, va_init, gr, vr_init)       
        # ligand
        va = F.leaky_relu(self.A(va), 0.1)  
        sa = self.sum_pool(ga, va)  
        va, sa = self.a_conv1(ga, va, ea, sa)
        va, sa = self.a_conv2(ga, va+va_init, ea, sa)
        fa = torch.cat((self.mean_pool(ga, va), sa), dim=-1)
        fa = self.B(fa)
        fa = fa + self.mean_pool(ga,va_init)    
        vr = F.leaky_relu(self.C(vr), 0.1)   
        sr = self.sum_pool(gr, vr)  
        vr, sr = self.r_conv1(gr, vr, torch.Tensor().reshape(gr.num_edges(),-1).to(device), sr)
        vr, sr = self.r_conv2(gr, vr+vr_init, torch.Tensor().reshape(gr.num_edges(),-1).to(device), sr)
        fr = torch.cat((self.mean_pool(gr, vr), sr), dim=-1)
        fr = self.D(fr)
        fr = fr + self.mean_pool(gr,vr_init)    
        # interaction
        vi = self.i_conf(gii, vi_init, ei)      
        vi = vi + vi_init    

        fi = self.mean_pool(gii, vi)   
        f = torch.cat((fa, fr, fi, vina), dim=-1)        
        y = self.classifier(f)

        return y

class MT(nn.Module):

    def __init__(self, in_dim):  
        super().__init__()

        self.A = nn.Linear(in_dim, 64) 
        self.B = nn.Linear(in_dim, 8)
        self.C = nn.Linear(64, in_dim)
        self.sum_pool = dglnn.SumPooling()
        self.D = nn.Linear(in_dim, 120)           
        self.E = nn.Linear(in_dim, 120)

    def forward(self, ga, va, gb, vb):        
        s = self.A(va)   
        h = self.B(va)  
        with ga.local_scope():    
            ga.ndata['h'] = h
            h = dgl.softmax_nodes(ga, 'h')  
            ga.ndata['h'] = h
            ga.ndata['s'] = s
            gga = dgl.unbatch(ga)  
            gp_ = torch.stack([torch.mm(g.ndata['s'].T, g.ndata['h']) for g in gga])  
            gp_ = self.C(gp_.mean(dim=-1)) 

        gp2_ = self.D(gp_)
        gp3_ = dgl.broadcast_nodes(gb, gp2_) 
        gp3_ = gp3_.permute(1,0)           
        r_ = torch.sum(torch.mm(self.E(vb),gp3_),dim=-1)
        pad_ = torch.sigmoid(r_)
        vbb = vb + vb * pad_.unsqueeze(1)

        return vbb

class SConv1(nn.Module):
    def __init__(self, v_dim, e_dim, h_dim, k_head):    
        super().__init__()

        self.A = nn.Linear(v_dim, h_dim)
        self.m2s = nn.ModuleList([SConv1.Helper(v_dim, h_dim) for _ in range(k_head)])
        self.B = nn.Linear(h_dim * k_head, h_dim)
        self.C = nn.Linear(v_dim, h_dim)
        self.D = nn.Linear(e_dim + v_dim, h_dim)   
        self.E = nn.Linear(h_dim + v_dim, h_dim)
        self.K = nn.Linear(e_dim, h_dim)       
        self.gate_update_m = SConv1.GateUpdate(h_dim)
        self.gate_update_s = SConv1.GateUpdate(h_dim)

    def __msg_func(self, edges):
        v = edges.src['v']
        e = edges.data['e']

        return {'ve': F.leaky_relu(self.K(e) * v,0.1)}   

    def forward(self, g, v, e, s):   
        s2s = torch.tanh(self.A(s)) 
        m2s = torch.cat([layer(g, v, s) for layer in self.m2s],dim=1)  
        m2s = torch.tanh(self.B(m2s))  
        s2m = torch.tanh(self.C(s))  
        s2m = dgl.broadcast_nodes(g, s2m)  

        with g.local_scope():
            g.ndata['v'] = v
            g.edata['e'] = e
            g.update_all(self.__msg_func, dglfn.sum('ve', 'sve'))
            svev = torch.cat((g.ndata['sve'], v),dim=1)   
        m2m = F.leaky_relu(self.E(svev), 0.1 )  
        vv = self.gate_update_m(m2m, s2m, v)  
        ss = self.gate_update_s(s2s, m2s, s) 

        return vv, ss

    class Helper(nn.Module):
        def __init__(self, v_dim, h_dim):
            super().__init__()

            self.A = nn.Linear(v_dim, h_dim) 
            self.B = nn.Linear(v_dim, h_dim) 
            self.C = nn.Linear(h_dim, 1)  
            self.D = nn.Linear(v_dim, h_dim) 

        def forward(self, g, v, s):
            d_node = torch.tanh(self.A(v)) 
            d_super = torch.tanh(self.B(s))  
            d_super = dgl.broadcast_nodes(g, d_super)

            a = self.C(d_node * d_super).reshape(-1)

            with g.local_scope():
                g.ndata['a'] = a
                a = dgl.softmax_nodes(g, 'a')  
    
                g.ndata['h'] = self.D(v) * a.unsqueeze(1)                 
                main2super_i = dgl.sum_nodes(g, 'h') 

            return main2super_i

    class GateUpdate(nn.Module): 

        def __init__(self, h_dim):
            super().__init__()

            self.A = nn.Linear(h_dim, h_dim)
            self.B = nn.Linear(h_dim, h_dim)
            self.gru = nn.GRUCell(h_dim, h_dim)  

        def forward(self, a, b, c):
            z = torch.sigmoid(self.A(a) + self.B(b))
            h = z * b + (1 - z) * a
            cc = self.gru(c, h)
            return cc

class SConvr1(nn.Module):
    def __init__(self, v_dim, e_dim, h_dim, k_head):    
        super().__init__()

        self.A = nn.Linear(v_dim, h_dim)
        self.m2s = nn.ModuleList([SConvr1.Helper(v_dim, h_dim) for _ in range(k_head)])
        self.B = nn.Linear(h_dim * k_head, h_dim)
        self.C = nn.Linear(v_dim, h_dim)
        self.D = nn.Linear(e_dim + v_dim, h_dim)
        self.E = nn.Linear(h_dim + v_dim, h_dim)
        self.gate_update_m = SConvr1.GateUpdate(h_dim)
        self.gate_update_s = SConvr1.GateUpdate(h_dim)

    def __msg_func(self, edges):
        v = edges.src['v']
        e = edges.data['e']
        return {'ve': F.leaky_relu(v,0.1)}
                 
    def forward(self, g, v, e, s):     
        s2s = torch.tanh(self.A(s))  
        m2s = torch.cat([layer(g, v, s) for layer in self.m2s],dim=1)  
        m2s = torch.tanh(self.B(m2s))  
        s2m = torch.tanh(self.C(s))  
        s2m = dgl.broadcast_nodes(g, s2m) 

        with g.local_scope():
            g.ndata['v'] = v
            g.edata['e'] = e
            g.update_all(self.__msg_func, dglfn.sum('ve', 'sve'))
            svev = torch.cat((g.ndata['sve'], v),dim=1)    
        m2m = F.leaky_relu(self.E(svev), 0.1 ) 
        vv = self.gate_update_m(m2m, s2m, v)
        ss = self.gate_update_s(s2s, m2s, s)

        return vv, ss

    class Helper(nn.Module):
        def __init__(self, v_dim, h_dim):
            super().__init__()
            self.A = nn.Linear(v_dim, h_dim)  
            self.B = nn.Linear(v_dim, h_dim)  
            self.C = nn.Linear(h_dim, 1)   
            self.D = nn.Linear(v_dim, h_dim) 

        def forward(self, g, v, s):
            d_node = torch.tanh(self.A(v))  
            d_super = torch.tanh(self.B(s))  
            d_super = dgl.broadcast_nodes(g, d_super)
            a = self.C(d_node * d_super).reshape(-1)

            with g.local_scope():
                g.ndata['a'] = a
                a = dgl.softmax_nodes(g, 'a')    
                g.ndata['h'] = self.D(v) * a.unsqueeze(1)    
                main2super_i = dgl.sum_nodes(g, 'h')

            return main2super_i 

    class GateUpdate(nn.Module):
        def __init__(self, h_dim):
            super().__init__()
            self.A = nn.Linear(h_dim, h_dim)
            self.B = nn.Linear(h_dim, h_dim)
            self.gru = nn.GRUCell(h_dim, h_dim)

        def forward(self, a, b, c):
            z = torch.sigmoid(self.A(a) + self.B(b))
            h = z * b + (1 - z) * a
            cc = self.gru(c, h)
            return cc
    
class DistanceConv(nn.Module):

    def __init__(self, dim, rc, depth):
        super().__init__()
        self.rs = nn.Parameter(torch.rand(1))
        self.sigma = nn.Parameter(torch.rand(1))
        self.A = nn.Linear(dim, dim)
        self.rc = rc
        self.depth = depth

    def f(self, r):
        return torch.exp((-torch.square(r - self.rs) / torch.square(self.sigma))) * \
               0.5 * torch.cos(np.pi * r / self.rc) * (r < self.rc)

    def __msg_func(self, edges):
        v = edges.src['v']
        f = edges.data['f']
        return {'vf': f * v}

    def forward(self, g, v, e):
        with g.local_scope():
            g.ndata['v'] = v
            g.edata['f'] = self.f(e)
            for _ in range(self.depth):
                g.update_all(self.__msg_func, dglfn.sum('vf', 'svf')) 
                g.ndata['v'] = torch.relu(self.A(g.ndata['svf']+v))
            v = g.ndata['v']

        return v
