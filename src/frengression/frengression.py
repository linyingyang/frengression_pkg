import torch
from engression.models import StoNet, StoNetBase
from engression.loss_func import energy_loss_two_sample
import numpy as np
import copy
sigmoid = torch.nn.Sigmoid()

class Frengression(torch.nn.Module):
    def __init__(self, x_dim, y_dim, z_dim,
                 num_layer=3, hidden_dim=100, noise_dim=10,
                 x_binary=False, 
                 #z_binary=False,
                 z_binary_dims = 0, # number of binary confounders.
                 y_binary=False,
                 device=torch.device('cuda')):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.x_binary = x_binary
        # self.z_binary = z_binary
        self.z_binary_dims = z_binary_dims
        self.y_binary = y_binary
        self.device = device
        self.model_xz = StoNet(0, x_dim + z_dim, num_layer, hidden_dim, max(x_dim + z_dim, noise_dim), add_bn=False, noise_all_layer=False).to(device)
        out_act = 'sigmoid' if y_binary else None
        self.model_y = StoNet(x_dim + y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act).to(device)
        self.model_eta = StoNet(x_dim + z_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False).to(device)
        
    def train_xz(self, x, z, num_iters=100, lr=1e-4, print_every_iter=10):
        self.model_xz.train()
        self.optimizer_xz = torch.optim.Adam(self.model_xz.parameters(), lr=lr)
        xz = torch.cat([x, z], dim=1)
        xz = xz.to(self.device)
        for i in range(num_iters):
            self.optimizer_xz.zero_grad()
            sample1 = self.model_xz(x.size(0))
            sample2 = self.model_xz(x.size(0))
            if self.x_binary:
                sample1[:, :self.x_dim] = sigmoid(sample1[:, :self.x_dim])
                sample2[:, :self.x_dim] = sigmoid(sample2[:, :self.x_dim])
            if self.z_binary_dims > 0:
                sample1[:, self.x_dim:(self.x_dim+self.z_binary_dims)] = sigmoid(sample1[:, self.x_dim:(self.x_dim+self.z_binary_dims)])
                sample2[:, self.x_dim:(self.x_dim+self.z_binary_dims)] = sigmoid(sample2[:, self.x_dim:(self.x_dim+self.z_binary_dims)])
            loss, loss1, loss2 = energy_loss_two_sample(xz, sample1, sample2)
            loss.backward()
            self.optimizer_xz.step()
            if (i == 0) or ((i + 1) % print_every_iter == 0):
                print(f'Epoch {i + 1}: loss {loss.item():.4f}, loss1 {loss1.item():.4f}, loss2 {loss2.item():.4f}')
    
    def train_y(self, x, z, y, num_iters=100, lr=1e-3, print_every_iter=10, tol=0.01):

        self.model_y.train()
        self.model_eta.train()
        self.optimizer_y = torch.optim.Adam(list(self.model_y.parameters()) + list(self.model_eta.parameters()), lr=lr)
        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)
        xz = torch.cat([x, z], dim=1)
        xz = xz.to(self.device)
        for i in range(num_iters):
            self.optimizer_y.zero_grad()
            eta1 = self.model_eta(xz)
            eta2 = self.model_eta(xz)
            y_sample1 = self.model_y(torch.cat([x, eta1], dim=1))
            y_sample2 = self.model_y(torch.cat([x, eta2], dim=1))
            loss_y, loss1_y, loss2_y = energy_loss_two_sample(y, y_sample1, y_sample2)
            eta_true = torch.randn(y.size(), device=self.device)
            # eta1 = self.model_eta(xz)
            # eta2 = self.model_eta(xz[torch.randperm(x.size(0))])
            xz_decouple1 = torch.cat([x, z[torch.randperm(z.size(0))]], dim=1)
            xz_decouple2 = torch.cat([x, z[torch.randperm(z.size(0))]], dim=1)
            eta1 = self.model_eta(xz_decouple1)
            eta2 = self.model_eta(xz_decouple2)
            loss_eta, loss1_eta, loss2_eta = energy_loss_two_sample(eta_true, eta1, eta2)

            if abs(loss2_y.item() - loss1_y.item()) < tol and \
                abs(loss2_eta.item() - loss1_eta.item()) < tol:
                print(f"Stopping at iter {i}: |Δy|={(loss2_y-loss1_y).item():.4e}, |Δη|={(loss2_eta-loss1_eta).item():.4e}")
                break
            loss = loss_y + loss_eta
            loss.backward()
            self.optimizer_y.step()
            if (i == 0) or ((i + 1) % print_every_iter == 0):
                print(f'Epoch {i + 1}: loss {loss.item():.4f},\tloss_y {loss_y.item():.4f}, {loss1_y.item():.4f}, {loss2_y.item():.4f},\tloss_eta {loss_eta.item():.4f}, {loss1_eta.item():.4f}, {loss2_eta.item():.4f}')
    
    @torch.no_grad()
    def predict_causal(self, x, target="mean", sample_size=100):
        self.eval()
        x = x.to(self.device)
        return self.model_y.predict(x, target, sample_size)

    @torch.no_grad()
    def sample_xz(self, sample_size=100):
        self.eval()
        xz = self.model_xz(sample_size)
        x = xz[:, :self.x_dim]
        z = xz[:, self.x_dim:]
        if self.x_binary:
            x = (x > 0).float()
        if self.z_binary_dims>0:
            z[:, :self.z_binary_dims] = (z[:, :self.z_binary_dims] > 0).float()

        return x, z    
        
    @torch.no_grad()
    def sample_joint(self, sample_size=100):
        self.eval()
        xz = self.model_xz(sample_size)
        x = xz[:, :self.x_dim]
        z = xz[:, self.x_dim:]
        if self.x_binary:
            x = (x > 0).float()
        if self.z_binary_dims>0:
            z[:, :self.z_binary_dims] = (z[:, :self.z_binary_dims] > 0).float()
        xz = torch.cat([x, z], dim=1)
        eta = self.model_eta(xz)
        y = self.model_y(torch.cat([x, eta], dim=1))
        return x, y, z

    @torch.no_grad()
    def sample_causal_margin(self, x, sample_size=100):
        self.eval()
        y = self.model_y.sample(x, sample_size = sample_size)
        return y
    
    @torch.no_grad()
    def predict_conditional(self, x, xz, sample_size=100):
        self.eval()
        xz = xz.to(self.device)
        eta = self.model_eta(xz)
        y = self.model_y.predict(torch.cat([x, eta], dim=1), sample_size = sample_size)
        return y
    

    def specify_causal(self, causal_margin):
        self.model_y = causal_margin
    
    def reset_y_models(self):
        self.model_y = StoNet(self.x_dim + self.y_dim, self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False).to(self.device)
        self.model_eta = StoNet(self.x_dim + self.z_dim, self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False).to(self.device)


class FrengressionSeq(torch.nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, T, s_dim,
                 num_layer=3, hidden_dim=100, noise_dim=10,
                 x_binary=False, z_binary=False, y_binary=False, s_binary_dims = 0,
                 s_in_predict=True,
                 device=torch.device('cuda')):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.s_dim = s_dim
        self.s_binary_dims = s_binary_dims
        self.T = T
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.x_binary = x_binary
        self.z_binary = z_binary
        self.y_binary = y_binary
        self.s_in_predict = s_in_predict
        self.device = device
        # for xz:
        # generate x0z0
        
        self.model_xz = [
            StoNet(s_dim, x_dim + z_dim, num_layer, hidden_dim, max(x_dim + z_dim, noise_dim), add_bn=False, noise_all_layer=False,verbose=False).to(device)
        ]
        #generate x1z1 to xTzT
        for t in range(T-1):
            self.model_xz.append(
                StoNet(s_dim + (x_dim + z_dim) * (t + 1), x_dim + z_dim, num_layer, hidden_dim, max(x_dim + z_dim+y_dim, noise_dim), add_bn=False, noise_all_layer=False,verbose=False).to(device)
            )
        
        out_act = 'sigmoid' if y_binary else None
       

        if self.s_in_predict:
            self.model_y = [
                StoNet(s_dim + x_dim + y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(device)
            ]
            # generate y1 onwards
            for t in range(1,T):
                self.model_y.append(
                    StoNet(s_dim + x_dim * (t+1)+ y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(device)
            )
        else:
            self.model_y = [
                StoNet(x_dim + y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(device)
            ]
            # generate y1 onwards
            for t in range(1,T):
                self.model_y.append(
                    StoNet(x_dim * (t+1)+ y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(device)
            )
        # for e
        if self.s_in_predict:
            self.model_e = [
                StoNet(s_dim, z_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, verbose=False).to(device)
            ]
            # generate y1 onwards
            for t in range(1,T):
                self.model_e.append(
                    StoNet(s_dim + x_dim * t, z_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, verbose=False).to(device)
                )
        else:
            self.model_e = [
                StoNet(x_dim, z_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, verbose=False).to(device)
            ]
            # generate y1 onwards
            for t in range(1,T):
                self.model_e.append(
                    StoNet(x_dim * (t+1), z_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, verbose=False).to(device)
            )

        # for eta:
        self.model_eta = [
            StoNet(s_dim + x_dim + z_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False,verbose=False).to(device)
        ]

        for t in range(1,T):
            self.model_eta.append(
                StoNet(s_dim+(x_dim + z_dim)*(t + 1), y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, verbose=False).to(device)
            )
    
    
    def sample_xz(self, s=None, x=None, z=None):
        xz = self.model_xz[0](s)
        x0 = xz[:, :self.x_dim]
        z0 = xz[:, self.x_dim:]
        if self.x_binary:
            x0 = (x0 > 0).float()
        if self.z_binary:
            z0 = (z0 > 0).float()
        x_all = [x0]
        z_all = [z0]
        for t in range(1,self.T):
            sxz_p = torch.cat([s, x[:,:(t*self.x_dim)], z[:,:(t*self.z_dim)]], dim=1)
            xz = self.model_xz[t](sxz_p)
            xt = xz[:, :self.x_dim]
            zt = xz[:, self.x_dim:]
            if self.x_binary:
                xt = (xt > 0).float()
            x_all.append(xt)
            z_all.append(zt)
        return torch.cat(x_all, dim=1), torch.cat(z_all, dim=1)

    def sample_eta(self, s = None, x=None, z=None):
        eta_all = []
        for t in range(self.T):
            sxz_p = torch.cat([s, x[:, :((t+1)*self.x_dim)], z[:,:((t+1)*self.z_dim)]], dim=1)
            etat = self.model_eta[t](sxz_p)
            eta_all.append(etat)
        return torch.cat(eta_all, dim=1)
    
    def sample_e(self, s = None, x = None):
        e_all = [self.model_e[0](s)]
        for t in range(1, self.T):
            e_all.append(self.model_e[t](torch.cat([s,x[:, :self.x_dim*t]],dim=1)))
        return torch.cat(e_all, dim=1)
    
    def sample_y(self, s = None, x=None, eta=None):
        y_all = []
        for t in range(self.T):
            if self.s_in_predict:
                sxeta_p = torch.cat([s, x[:,:((t+1)*self.x_dim)], eta[:, (t*self.y_dim):((t+1)*self.y_dim)]], dim=1)
                yt = self.model_y[t](sxeta_p)
            else:
                xeta_p = torch.cat([x[:,:((t+1)*self.x_dim)], eta[:, (t*self.y_dim):((t+1)*self.y_dim)]], dim=1)
                yt = self.model_y[t](xeta_p)
            y_all.append(yt)
        return torch.cat(y_all, dim=1)

        

    def train_xz(self, s, x, z, num_iters=100, lr=1e-3, print_every_iter=10):
        for model in self.model_xz:
            model.train()
        all_parameters = []
        for t in range(self.T):
            all_parameters += list(self.model_xz[t].parameters())
        self.optimizer_xz = torch.optim.Adam(all_parameters, lr=lr)
        xz = torch.cat([x, z], dim=1)
        for i in range(num_iters):
            self.optimizer_xz.zero_grad()
            sample1_x, sample1_z = self.sample_xz(s=s, x=x, z=z)
            sample1 = torch.cat([sample1_x, sample1_z], dim=1)
            sample2_x, sample2_z = self.sample_xz(s=s, x=x, z=z)
            sample2 = torch.cat([sample2_x, sample2_z], dim=1)
            if self.x_binary:
                sample1[:, :self.x_dim] = sigmoid(sample1[:, :self.x_dim])
                sample2[:, :self.x_dim] = sigmoid(sample2[:, :self.x_dim])
            if self.z_binary:
                sample1[:, self.x_dim:] = sigmoid(sample1[:, self.x_dim:])
                sample2[:, self.x_dim:] = sigmoid(sample2[:, self.x_dim:])
            loss, loss1, loss2 = energy_loss_two_sample(xz, sample1, sample2)
            loss.backward()
            self.optimizer_xz.step()
            if (i == 0) or ((i + 1) % print_every_iter == 0):
                print(f'Epoch {i + 1}: loss {loss.item():.4f}, loss1 {loss1.item():.4f}, loss2 {loss2.item():.4f}')

    def train_e(self, s, x, z, num_iters = 100, lr=1e-3, print_every_iter=10):
        for model in self.model_e:
            model.train()
        all_parameters = []
        for t in range(self.T):
            all_parameters += list(self.model_e[t].parameters())
        self.optimizer_e = torch.optim.Adam(all_parameters, lr=lr)
        for i in range(num_iters):
            self.optimizer_e.zero_grad()
            sample1_e = self.sample_e(s=s, x=x)
            sample2_e = self.sample_e(s=s, x=x)
            
            loss, loss1, loss2 = energy_loss_two_sample(z, sample1_e, sample2_e)
            loss.backward()
            self.optimizer_e.step()
            if (i == 0) or ((i + 1) % print_every_iter == 0):
                print(f'Epoch {i + 1}: loss {loss.item():.4f}, loss1 {loss1.item():.4f}, loss2 {loss2.item():.4f}')

    
    def train_y(self, s, x, z, y, num_iters=100, lr=1e-4, print_every_iter=10):
        all_parameters = []
        for t in range(self.T):
            self.model_y[t].train()
            self.model_eta[t].train()
            all_parameters += list(self.model_y[t].parameters())+ list(self.model_eta[t].parameters())

        self.optimizer_y = torch.optim.Adam(all_parameters, lr=lr)
        s = s.to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)


        for i in range(num_iters):
            
            self.optimizer_y.zero_grad()
            eta1 = self.sample_eta(s=s, x=x,z=z)
            eta2 = self.sample_eta(s=s, x=x,z=z)
            y_sample1 = self.sample_y(s=s,x=x, eta = eta1)
            y_sample2 = self.sample_y(s=s,x=x, eta = eta2)
            loss_y, loss1_y, loss2_y = energy_loss_two_sample(y, y_sample1, y_sample2)
            
            eta_true = torch.randn(y.size(), device=self.device)

            # perm = torch.randperm(x.size(0))
            z1 = self.sample_e(s=s, x=x)
            z2 = self.sample_e(s=s, x=x)
            eta1 = self.sample_eta(s=s, x=x, z=z1)
            eta2 = self.sample_eta(s=s, x=x,z=z2)
            loss_eta, loss1_eta, loss2_eta = energy_loss_two_sample(eta_true, eta1, eta2)
            loss = loss_y + loss_eta
            loss.backward()
            self.optimizer_y.step()
            if (i == 0) or ((i + 1) % print_every_iter == 0):
                print(f'Epoch {i + 1}: loss {loss.item():.4f},\tloss_y {loss_y.item():.4f}, {loss1_y.item():.4f}, {loss2_y.item():.4f},\tloss_eta {loss_eta.item():.4f}, {loss1_eta.item():.4f}, {loss2_eta.item():.4f}')
    
    @torch.no_grad()
    def predict_causal(self, s, x, target="mean", sample_size=100):
        self.eval()
        x = x.to(self.device)
        s = s.to(self.device)
        all_y = []
        for t in range(self.T):
            if self.s_in_predict:
                yt = self.model_y[t].predict(torch.cat([s,x[:,:(t+1)*self.x_dim]],dim=1), target, sample_size)
            else:
                yt = self.model_y[t].predict(x[:,:(t+1)*self.x_dim], target, sample_size)
            all_y.append(yt)
        return all_y
    
    @torch.no_grad()
    def sample_causal_margin(self,s, x, sample_size=100):
        self.eval()
        x = x.to(self.device)
        s = s.to(self.device)
        all_y = []
        for t in range(self.T):
            if self.s_in_predict:
                yt = self.model_y[t].sample(torch.cat([s,x[:,:(t+1)*self.x_dim]], dim=1), sample_size = sample_size)
            else:
                yt = self.model_y[t].sample(x[:,:(t+1)*self.x_dim], sample_size = sample_size)
            all_y.append(yt)
        return all_y

    @torch.no_grad()
    def sample_joint(self, s, sample_size=100):
        self.eval()
        s = s.to(self.device)

        x_all, z_all = self.sample_xz(s=s)

        eta_all = self.sample_eta(s=s, x=x_all, z=z_all)

        y_all = self.sample_y(s=s, x=x_all, eta=eta_all)

        if self.y_binary:
            y_all = (y_all > 0.5).int()

        return x_all, z_all, y_all

    def reset_y_models(self):
        if self.s_in_predict:
            self.model_y = [
                StoNet(self.s_dim + self.x_dim + self.y_dim, self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(self.device)
            ]
            # generate y1 onwards
            for t in range(1,self.T):
                self.model_y.append(
                    StoNet(self.s_dim + self.x_dim * (t+1)+ self.y_dim, self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(self.device)
            )
        else:
            self.model_y = [
                StoNet(self.x_dim + self.y_dim, self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(self.device)
            ]
            # generate y1 onwards
            for t in range(1,self.T):
                self.model_y.append(
                    StoNet(self.x_dim * (t+1)+ self.y_dim, self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(self.device)
            )

        # for eta:
        self.model_eta = [
            StoNet(self.s_dim + self.x_dim + self.z_dim, self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False,verbose=False).to(self.device)
        ]

        for t in range(1,self.T):
            self.model_eta.append(
                StoNet(self.s_dim+(self.x_dim + self.z_dim)*(t + 1), self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False, verbose=False).to(self.device)
            )
    

class FrengressionSurv(torch.nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, T, s_dim,
                 num_layer=3, hidden_dim=100, noise_dim=10,
                 x_binary=False, z_binary=False, y_binary=True,
                 s_in_predict = True,
                 device=torch.device('cuda')):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.s_dim = s_dim
        self.T = T
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.x_binary = x_binary
        self.z_binary = z_binary
        self.y_binary = y_binary # Default setting of survivl data : binary outcome
        self.device = device
        self.s_in_predict=s_in_predict
        # for xz:
        # generate x0z0
        self.model_xz = [
            StoNet(s_dim, x_dim + z_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False,verbose=False).to(device)
        ]
        #generate x1z1 to xTzT
        for t in range(T-1):
            self.model_xz.append(
                StoNet(s_dim + (x_dim + z_dim)* (t + 1) , x_dim + z_dim, num_layer, hidden_dim, max(x_dim + z_dim+y_dim, noise_dim), add_bn=False, noise_all_layer=False,verbose=False).to(device)
            )

        
        # for e
        if self.s_in_predict:
            self.model_e = [
                StoNet(s_dim, z_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, verbose=False).to(device)
            ]
            # generate y1 onwards
            for t in range(1,T):
                self.model_e.append(
                    StoNet(s_dim + x_dim * t, z_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, verbose=False).to(device)
                )
        else:
            self.model_e = [
                StoNet(x_dim, z_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, verbose=False).to(device)
            ]
            # generate y1 onwards
            for t in range(1,T):
                self.model_e.append(
                    StoNet(x_dim * (t+1), z_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, verbose=False).to(device)
        )
        
        out_act = 'sigmoid' if y_binary else None
       
        # for y
        # generate y0
        if self.s_in_predict:
            self.model_y = [
                StoNet(s_dim + x_dim + y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(device)
            ]
            # generate y1 onwards
            for t in range(1,T):
                self.model_y.append(
                    StoNet(s_dim + x_dim * (t+1)+ y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(device)
                    # StoNet((x_dim+y_dim) * (t+1), y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act).to(device)
                )
        else:
            self.model_y = [
                StoNet(x_dim + y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(device)
            ]
            # generate y1 onwards
            for t in range(1,T):
                self.model_y.append(
                    StoNet(x_dim * (t+1)+ y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(device)
                )
        
        # for eta:
        self.model_eta = [
            StoNet(s_dim + x_dim + z_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False,verbose=False).to(device)
        ]

        for t in range(1,T):
            self.model_eta.append(
                StoNet(s_dim+(x_dim + z_dim)*(t + 1), y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, verbose=False).to(device)
            )
        
    
    def sample_xz(self, s=None, x=None, z=None, y = None):
        xz = self.model_xz[0](s)
        x0 = xz[:, :self.x_dim]
        z0 = xz[:, self.x_dim:]
        if self.x_binary:
            x0 = (x0 > 0).float()
        if self.z_binary:
            z0 = (z0 > 0).float()
        x_all = [x0]
        z_all = [z0]
        for t in range(1,self.T):
            sxz_p = torch.cat([s, x[:,:(t*self.x_dim)], z[:,:(t*self.z_dim)]], dim=1)
            xz = self.model_xz[t](sxz_p)
            xt = xz[:, :self.x_dim]
            zt = xz[:, self.x_dim:]
            if self.x_binary:
                xt = (xt > 0).float()
            if self.z_binary:
                zt = (zt > 0).float()
            x_all.append(xt)
            z_all.append(zt)
        

        return torch.cat(x_all, dim=1), torch.cat(z_all, dim=1)

        

    def train_xz(self, s, x, z, y, num_iters=100, lr=1e-3, print_every_iter=10):
        all_parameters = []
        for t in range(self.T):
            self.model_xz[t].train()
            all_parameters += list(self.model_xz[t].parameters())
        
        self.optimizer_xz = torch.optim.Adam(all_parameters, lr=lr)
        s = s.to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)

        n = x.shape[0]

        for i in range(num_iters):
            event_indicator = (y>0).float()
            c = torch.cumsum(event_indicator, dim=1)
            c_shifted = torch.zeros_like(c)
            c_shifted[:, 1:] = c[:, :-1]
            mask = (c_shifted > 0)
            y_masked = y.clone()
            y_masked[mask] = -1

            x_list = [x[:,:self.x_dim].clone()]
            s_list = [s[:,:self.s_dim].clone()]
            z_list = [z[:,:self.z_dim].clone()]

            x_label_list = [x[:,:self.x_dim].clone()]
            z_label_list = [z[:,:self.z_dim].clone()]
                        # resample from data
            for t in range(1, self.T):
                # Determine valid and invalid indices for the current time step t
                valid_idx = (y_masked[:, t] >= -0.5).nonzero(as_tuple=True)[0]
                invalid_idx = (y_masked[:, t] < -0.5).nonzero(as_tuple=True)[0]

                if invalid_idx.numel() == 0:
                    x_list.append(x[:, :(t*self.x_dim)].clone())
                    z_list.append(z[:, :(t*self.z_dim)].clone())
                    s_list.append(s[:, :self.s_dim].clone())
                    x_label_list.append(x[:, (t*self.x_dim):((t+1)*self.x_dim)].clone())
                    z_label_list.append(z[:, (t*self.z_dim):((t+1)*self.z_dim)].clone())
                else:
                    # For each invalid position, sample a replacement index from the valid positions.
                    # The number of samples equals the number of invalid positions.
                    sampled_idx = valid_idx[torch.randint(0, len(valid_idx), (len(invalid_idx),))]
                    
                    # For the current time step, get a copy of the original data.
                    # This ensures that valid positions remain unchanged.
                    x_t = x[:, :((t)*self.x_dim)].clone()
                    z_t = z[:, :((t)*self.z_dim)].clone()
                    s_t = s[:, :self.s_dim].clone()

                    x_t_label = x[:, (t*self.x_dim):(t+1)*self.x_dim].clone()
                    z_t_label = z[:, (t*self.z_dim):(t+1)*self.z_dim].clone()
                    
                    # Replace only the invalid positions with the sampled valid ones.
                    x_t[invalid_idx] = x[sampled_idx, :(t*self.x_dim)].clone()
                    z_t[invalid_idx] = z[sampled_idx, :(t*self.z_dim)].clone()
                    s_t[invalid_idx] = s[sampled_idx, :self.s_dim].clone()

                    x_t_label[invalid_idx] = x[sampled_idx, (t*self.x_dim):(t+1)*self.x_dim].clone()
                    z_t_label[invalid_idx] = z[sampled_idx, (t*self.z_dim):(t+1)*self.z_dim].clone()
                    
                    # Append the corrected data for time step t.

                    x_list.append(x_t)
                    z_list.append(z_t)
                    s_list.append(s_t)

                    x_label_list.append(x_t_label)
                    z_label_list.append(z_t_label)
            

            x_sample = torch.cat(x_label_list, dim=1)
            z_sample = torch.cat(z_label_list, dim=1)
            xz_sample = torch.cat([x_sample, z_sample], dim=1)
            
            self.optimizer_xz.zero_grad()
            xz_sample1 = self.model_xz[0](s)
            xz_sample2 = self.model_xz[0](s)
            if self.x_binary:
                xz_sample1[:, :self.x_dim] =  sigmoid(xz_sample1[:, :self.x_dim])
                xz_sample2[:, :self.x_dim] = sigmoid(xz_sample2[:, :self.x_dim])
            if self.z_binary:
                xz_sample1[:, self.x_dim:] = sigmoid(xz_sample1[:, self.x_dim:])
                xz_sample2[:, self.x_dim:] = sigmoid(xz_sample2[:, self.x_dim:])
            x_sample1 =  [xz_sample1[:, :self.x_dim]]
            z_sample1 =  [xz_sample1[:,self.x_dim:]]
            
            x_sample2 = [xz_sample2[:, :self.x_dim]]
            z_sample2 = [xz_sample2[:,self.x_dim:]]

            for t in range(1, self.T):
                sxz_p = torch.cat([s_list[t],x_list[t], z_list[t]],dim=1)
                xz_sample1 = self.model_xz[t](sxz_p)
                xz_sample2 = self.model_xz[t](sxz_p)
                if self.x_binary:
                    xz_sample1[:, :self.x_dim] = sigmoid(xz_sample1[:, :self.x_dim])
                    xz_sample2[:, :self.x_dim] = sigmoid(xz_sample2[:, :self.x_dim])
                if self.z_binary:
                    xz_sample1[:, self.x_dim:] = sigmoid(xz_sample1[:, self.x_dim:])
                    xz_sample2[:, self.x_dim:] = sigmoid(xz_sample2[:, self.x_dim:])
                x_sample1.append(xz_sample1[:, :self.x_dim])
                z_sample1.append(xz_sample1[:, self.x_dim:])
                x_sample2.append(xz_sample2[:, :self.x_dim])
                z_sample2.append(xz_sample2[:, self.x_dim:])
            
            x_sample1 = torch.cat(x_sample1, dim=1)
            z_sample1 = torch.cat(z_sample1, dim=1)
            xz_sample1 = torch.cat([x_sample1, z_sample1],dim=1)

            x_sample2 = torch.cat(x_sample2, dim=1)
            z_sample2 = torch.cat(z_sample2, dim=1)
            xz_sample2 = torch.cat([x_sample2, z_sample2],dim=1)

            loss, loss1, loss2 = energy_loss_two_sample(xz_sample, xz_sample1, xz_sample2)
            loss.backward()
            self.optimizer_xz.step()
            if (i == 0) or ((i + 1) % print_every_iter == 0):
                print(f'Epoch {i + 1}: loss {loss.item():.4f}, loss1 {loss1.item():.4f}, loss2 {loss2.item():.4f}')

    def train_e(self, s, x, z, y, num_iters=100, lr=1e-3, print_every_iter=10):
        all_parameters = []
        for t in range(self.T):
            self.model_e[t].train()
            all_parameters += list(self.model_e[t].parameters())
        
        self.optimizer_e = torch.optim.Adam(all_parameters, lr=lr)
        s = s.to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)

        n = x.shape[0]

        for i in range(num_iters):
            event_indicator = (y>0).float()
            c = torch.cumsum(event_indicator, dim=1)
            c_shifted = torch.zeros_like(c)
            c_shifted[:, 1:] = c[:, :-1]
            mask = (c_shifted > 0)
            y_masked = y.clone()
            y_masked[mask] = -1

            x_list = [x[:,:self.x_dim].clone()]
            s_list = [s[:,:self.s_dim].clone()]
            z_list = [z[:,:self.z_dim].clone()]

            x_label_list = [x[:,:self.x_dim].clone()]
            z_label_list = [z[:,:self.z_dim].clone()]
                        # resample from data
            for t in range(1, self.T):
                # Determine valid and invalid indices for the current time step t
                valid_idx = (y_masked[:, t] >= -0.5).nonzero(as_tuple=True)[0]
                invalid_idx = (y_masked[:, t] < -0.5).nonzero(as_tuple=True)[0]

                if invalid_idx.numel() == 0:
                    x_list.append(x[:, :(t*self.x_dim)].clone())
                    z_list.append(z[:, :(t*self.z_dim)].clone())
                    s_list.append(s[:, :self.s_dim].clone())
                    x_label_list.append(x[:, (t*self.x_dim):((t+1)*self.x_dim)].clone())
                    z_label_list.append(z[:, (t*self.z_dim):((t+1)*self.z_dim)].clone())
                else:
                    # For each invalid position, sample a replacement index from the valid positions.
                    # The number of samples equals the number of invalid positions.
                    sampled_idx = valid_idx[torch.randint(0, len(valid_idx), (len(invalid_idx),))]
                    
                    # For the current time step, get a copy of the original data.
                    # This ensures that valid positions remain unchanged.
                    x_t = x[:, :((t)*self.x_dim)].clone()
                    z_t = z[:, :((t)*self.z_dim)].clone()
                    s_t = s[:, :self.s_dim].clone()

                    x_t_label = x[:, (t*self.x_dim):(t+1)*self.x_dim].clone()
                    z_t_label = z[:, (t*self.z_dim):(t+1)*self.z_dim].clone()
                    
                    # Replace only the invalid positions with the sampled valid ones.
                    x_t[invalid_idx] = x[sampled_idx, :(t*self.x_dim)].clone()
                    z_t[invalid_idx] = z[sampled_idx, :(t*self.z_dim)].clone()
                    s_t[invalid_idx] = s[sampled_idx, :self.s_dim].clone()

                    x_t_label[invalid_idx] = x[sampled_idx, (t*self.x_dim):(t+1)*self.x_dim].clone()
                    z_t_label[invalid_idx] = z[sampled_idx, (t*self.z_dim):(t+1)*self.z_dim].clone()
                    
                    # Append the corrected data for time step t.

                    x_list.append(x_t)
                    z_list.append(z_t)
                    s_list.append(s_t)

                    x_label_list.append(x_t_label)
                    z_label_list.append(z_t_label)

            z_sample = torch.cat(z_label_list, dim=1)

            self.optimizer_e.zero_grad()
            z_sample1 = [self.model_e[0](s)]
            z_sample2 = [self.model_e[0](s)]
            if self.z_binary:
                z_sample1[:,:] = sigmoid(z_sample1[:,:])
                z_sample2[:,:] = sigmoid(z_sample2[:,:])


            for t in range(1, self.T):
                sx_p = torch.cat([s_list[t],x_list[t][:,:(self.x_dim*t)]],dim=1)
                zp_sample1 = self.model_e[t](sx_p)
                zp_sample2 = self.model_e[t](sx_p)
                if self.z_binary:
                    zp_sample1[:,:] = sigmoid(zp_sample1[:,:])
                    zp_sample2[:,:] = sigmoid(zp_sample2[:,:])

                z_sample1.append(zp_sample1)
                z_sample2.append(zp_sample2)
            
            z_sample1 = torch.cat(z_sample1, dim=1)
            z_sample2 = torch.cat(z_sample2, dim=1)

            loss, loss1, loss2 = energy_loss_two_sample(z_sample, z_sample1, z_sample2)
            loss.backward()
            self.optimizer_e.step()
            if (i == 0) or ((i + 1) % print_every_iter == 0):
                print(f'Epoch {i + 1}: loss {loss.item():.4f}, loss1 {loss1.item():.4f}, loss2 {loss2.item():.4f}')
    
    
    def sample_e(self, s = None, x = None, T= None):
        e_all = [self.model_e[0](s)]
        for t in range(1, T):
            e_all.append(self.model_e[t](torch.cat([s,x[:, :(self.x_dim*t)]],dim=1)))
        return torch.cat(e_all, dim=1)

    
    def train_y(self, s, x, z, y, num_iters=100, lr=1e-3, print_every_iter=10, reg_lambda = 0):
        all_parameters = []
        for t in range(self.T):
            self.model_y[t].train()
            self.model_eta[t].train()
            all_parameters += list(self.model_y[t].parameters())+ list(self.model_eta[t].parameters())

        self.optimizer_y = torch.optim.Adam(all_parameters, lr=lr)
        s = s.to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)

        n = x.shape[0]
        num_events = torch.nansum(y == 1).item()
        event_ratio_true = num_events / n

        
        for i in range(num_iters):
            event_indicator = (y>0).float()
            c = torch.cumsum(event_indicator, dim=1)
            c_shifted = torch.zeros_like(c)
            c_shifted[:, 1:] = c[:, :-1]
            mask = (c_shifted > 0)
            y_masked = y.clone()
            y_masked[mask] = -1


            y_list = [y[:,:self.y_dim].clone()]
            x_list = [x[:,:self.x_dim].clone()]
            s_list = [s[:,:self.s_dim].clone()]
            z_list = [z[:,:self.z_dim].clone()]

            # resample from data
            for t in range(1, self.T):

                valid_idx = (y_masked[:, t] >= -0.5).nonzero(as_tuple=True)[0]
                invalid_idx = (y_masked[:, t] < -0.5).nonzero(as_tuple=True)[0]

                if invalid_idx.numel() == 0:
                    y_list.append(y[:, (t*self.y_dim):((t+1)*self.y_dim)].clone())
                    x_list.append(x[:, :((t+1)*self.x_dim)].clone())
                    z_list.append(z[:, :((t+1)*self.z_dim)].clone())
                    s_list.append(s[:,:self.s_dim].clone())
                else:
                    # For each invalid position, sample a replacement index from the valid positions.
                    # The number of samples equals the number of invalid positions.
                    sampled_idx = valid_idx[torch.randint(0, len(valid_idx), (len(invalid_idx),))]
                    
                    # For the current time step, get a copy of the original data.
                    # This ensures that valid positions remain unchanged.
                    y_t = y[:, (t*self.y_dim):((t+1)*self.y_dim)].clone()
                    x_t = x[:, :((t+1)*self.x_dim)].clone()
                    z_t = z[:, :((t+1)*self.z_dim)].clone()
                    s_t = s[:, :self.s_dim].clone()
                    
                    # Replace only the invalid positions with the sampled valid ones.
                    y_t[invalid_idx] = y[sampled_idx, (t*self.y_dim):((t+1)*self.y_dim)].clone()
                    x_t[invalid_idx] = x[sampled_idx, :((t+1)*self.x_dim)].clone()
                    z_t[invalid_idx] = z[sampled_idx, :((t+1)*self.z_dim)].clone()
                    s_t[invalid_idx] = s[sampled_idx, :self.s_dim].clone()
                    
                    # Append the corrected data for time step t.
                    y_list.append(y_t)
                    x_list.append(x_t)
                    z_list.append(z_t)
                    s_list.append(s_t)
            y_sample = torch.cat(y_list, dim=1)

            self.optimizer_y.zero_grad()
            y_sample1 = []
            y_sample2 = []
            for t in range(self.T):
                sxz_p = torch.cat([s_list[t], x_list[t], z_list[t]], dim=1)
                
                etat1 = self.model_eta[t](sxz_p)
                etat2 = self.model_eta[t](sxz_p)
                if self.s_in_predict:
                    sxeta_p1 = torch.cat([s_list[t], x_list[t], etat1], dim=1)
                    yt1 = self.model_y[t](sxeta_p1)
                    sxeta_p2 = torch.cat([s_list[t], x_list[t], etat2], dim=1)
                    yt2 = self.model_y[t](sxeta_p2)
                else:
                    xeta_p1 = torch.cat([x_list[t], etat1], dim=1)
                    xeta_p2 = torch.cat([x_list[t], etat2], dim=1)
                    yt1 = self.model_y[t](xeta_p1)
                    yt2 = self.model_y[t](xeta_p2)
                y_sample1.append(yt1)
                y_sample2.append(yt2)

            y_sample1_cat = torch.cat(y_sample1,dim=1)
            y_sample2_cat = torch.cat(y_sample2,dim=1)
            loss_y, loss1_y, loss2_y = energy_loss_two_sample(y_sample, y_sample1_cat, y_sample2_cat)
            
            eta_true_list = []
            for t in range(self.T):
                # sxz_p = torch.cat([s_list[t], x_list[t], z_list[t]], dim=1)
                eta_true_t = torch.randn(n, self.y_dim, device=self.device)
                eta_true_list.append(eta_true_t)
                
            eta_true = torch.cat(eta_true_list, dim=1)

            # sample z|pa(x) first
            z_sample_list1 = [self.model_e[0](s_list[0])]
            z_sample_list2 = [self.model_e[0](s_list[0])]

            for t in range(1,self.T):
                z_sample_list1.append(self.sample_e(s=s_list[t], x=x_list[t][:, :(self.x_dim*t)], T=t))
                z_sample_list2.append(self.sample_e(s=s_list[t], x=x_list[t][:, :(self.x_dim*t)], T=t))

            eta1 = []
            eta2 = []
            for t in range(self.T):
                # perm = torch.randperm(x.size(0))
                # sxz_p1 = torch.cat([s_list[t], x_list[t], z_list[t]], dim=1)
                # sxz_p2 = torch.cat([s_list[t][perm], x_list[t][perm], z_list[t][perm]], dim=1)

                sxz_p1 = torch.cat([s_list[t], x_list[t], z_sample_list1[t]], dim=1)
                sxz_p2 = torch.cat([s_list[t], x_list[t], z_sample_list2[t]], dim=1)
                etat1 = self.model_eta[t](sxz_p1)
                etat2 = self.model_eta[t](sxz_p2)
 
                eta1.append(etat1)
                eta2.append(etat2)
            eta1_cat = torch.cat(eta1,dim=1)
            eta2_cat = torch.cat(eta2,dim=1)

            loss_eta, loss1_eta, loss2_eta = energy_loss_two_sample(eta_true, eta1_cat, eta2_cat)
            ##
            # Convert all to float and compute per-dimension means
            mean_y_sample = y_sample.float().mean(dim=0)  # shape: [d]
            mean_y_sample1 = y_sample1_cat.float().mean(dim=0)  # shape: [d]
            mean_y_sample2 = y_sample2_cat.float().mean(dim=0)  # shape: [d]

            # Avoid division by zero by adding a small epsilon
            eps = 1e-6

            # Compute squared relative error per dimension and sum
            marginal_loss = (((mean_y_sample1 / (mean_y_sample + eps)) - 1) ** 2).mean() + (((mean_y_sample2 / (mean_y_sample + eps)) - 1) ** 2).mean()


            
            loss = loss_y + loss_eta + reg_lambda * marginal_loss
            # loss = reg_lambda * marginal_loss
            loss.backward()
            self.optimizer_y.step()
            if (i == 0) or ((i + 1) % print_every_iter == 0):
                print(f'Epoch {i + 1}: loss {loss.item():.4f},\tloss_y {loss_y.item():.4f}, {loss1_y.item():.4f}, {loss2_y.item():.4f},\tloss_eta {loss_eta.item():.4f}, {loss1_eta.item():.4f}, {loss2_eta.item():.4f}, \tmarginal_loss {marginal_loss.item():.4f}')
                print(f'Epoch {i + 1}: y_sample_mean {y_sample.float().mean()}')
                print(f'Epoch {i + 1}: y_sample1_cat.float().mean() {y_sample1_cat.float().mean()}')


    
    @torch.no_grad()
    def sample_causal_margin(self,s, x, sample_size=100):
        self.eval()
        x = x.to(self.device)
        s = s.to(self.device)
        all_y = []
        for t in range(self.T):
            if self.s_in_predict:
                yt = ((self.model_y[t].sample(torch.cat([s,x[:,:(t+1)*self.x_dim]], dim=1), sample_size = sample_size))>0.5).float()
            else:
                yt = ((self.model_y[t].sample(x[:,:(t+1)*self.x_dim], sample_size = sample_size))>0.5).float()
            all_y.append(yt)

        return torch.cat(all_y,dim=1).permute(2,0,1).squeeze(0)

    @torch.no_grad()
    def sample_joint(self,s, sample_size=100):
        self.eval()
        s = s.to(self.device)
        xz = self.model_xz[0](s)
        x0 = xz[:, :self.x_dim]
        z0 = xz[:, self.x_dim:]
        if self.x_binary:
            x0 = (x0 > 0).float()
        if self.z_binary:
            z0 = (z0 > 0).float()
        x_all = x0
        z_all = z0
        sxz_p = torch.cat([s, x0, z0], dim=1)
        etat0 = self.model_eta[0](sxz_p)
        sxeta_p0 = torch.cat([s, x0, etat0], dim=1)
        y0 = (self.model_y[0](sxeta_p0)>0.5).int()
        y_all = y0
        for t in range(1,self.T):
            sxzy_p = torch.cat([s, x_all[:,:(t*self.x_dim)], z_all[:,:(t*self.z_dim)]], dim=1)
            xz = self.model_xz[t](sxzy_p)
            xt = xz[:, :self.x_dim]
            zt = xz[:, self.x_dim:]
            if self.x_binary:
                xt = (xt > 0).float()
            if self.z_binary:
                zt = (zt > 0).float()
            x_all = torch.cat([x_all, xt], dim=1)
            z_all = torch.cat([z_all, zt], dim=1)

            sxz_p = torch.cat([s, x_all, z_all], dim=1)
            etat = self.model_eta[t](sxz_p)
            sxeta_p = torch.cat([s, x_all, etat], dim=1)
            yt = (self.model_y[t](sxeta_p)>0.5).int()
            y_all = torch.cat([y_all, yt], dim=1)
        
        return x_all, z_all, y_all

   
    @torch.no_grad()
    def predict_causal(self, s, x, sample_size=100):
        y_causal_margin = self.sample_causal_margin(s, x, sample_size)
        event_indicator = (y_causal_margin>0).float()
        
        return event_indicator
        
    def reset_y_models(self):
    # for y
        # generate y0
        if self.s_in_predict:
            self.model_y = [
                StoNet(self.s_dim + self.x_dim + self.y_dim, self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(self.device)
            ]
            # generate y1 onwards
            for t in range(1,self.T):
                self.model_y.append(
                    StoNet(self.s_dim + self.x_dim * (t+1)+ self.y_dim, self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(self.device)
                    # StoNet((x_dim+y_dim) * (t+1), y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act).to(device)
                )
        else:
            self.model_y = [
                StoNet(self.x_dim + self.y_dim, self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(self.device)
            ]
            # generate y1 onwards
            for t in range(1, self.T):
                self.model_y.append(
                    StoNet(self.x_dim * (t+1)+ self.y_dim, self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(self.device)
                )
        
        # for eta:
        self.model_eta = [
            StoNet(self.s_dim + self.x_dim + self.z_dim, self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False,verbose=False).to(self.device)
        ]

        for t in range(1,T):
            self.model_eta.append(
                StoNet(self.s_dim+(self.x_dim + self.z_dim)*(t + 1), self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False, verbose=False).to(self.device)
            )
