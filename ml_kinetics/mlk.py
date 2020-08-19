import sympy
import torch
torch.manual_seed(12)
torch.cuda.manual_seed(12)
torch.cuda.manual_seed_all

dtype = torch.float
device = torch.device("cuda:0")
#device = torch.device("cpu")
import torch.nn as nn
from torch.autograd import grad
from torch.autograd import Variable
import numpy as np
from sklearn.linear_model import Lasso

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden_layers, out_size, scalers=None):
        super(Model, self).__init__()
        if scalers == None:
            self.scalers = torch.ones((1,input_size),device=device, dtype=dtype)
        else:
            self.scalers = scalers
        activation = nn.Tanh()
        #activation = nn.Softplus(beta=1)
        layers =  [nn.Linear(input_size,hidden_size,bias=True),activation]
        for i in range(n_hidden_layers):
            layers += [nn.Linear(hidden_size,hidden_size,bias=False), activation]
        layers += [nn.Linear(hidden_size, out_size,bias=True),]
        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x*self.scalers)

def getdictionary(model,inputs,dictionary,sorted_free_symbols):
    """
    sorted_free_symbols...... list of sympy symbols sorted by length
    """
    u = model(torch.cat(list(inputs.values()),dim=1))
    terms = inputs.copy()
    terms['u'] = u
    for symbol in sorted_free_symbols:
        if str(symbol) in terms.keys():
            continue
        
        if len(str(symbol)) < 3:
            raise ValueError("Symbol '"+str(symbol)+"' illegal")
        for i in range(2,len(str(symbol))):
            key = str(symbol)[:(i+1)]
            if key not in terms.keys():
                if i == 2:
                    key_m = str(symbol)[0] 
                else:
                    key_m = str(symbol)[:i]
                func = terms[key_m]
                var = inputs[str(symbol)[i]]
                terms[key] = grad(func,var,create_graph=True,
                                   grad_outputs=torch.ones_like(func))[0]
    M =  torch.cat(eval(dictionary,terms),dim=1)
    u_t = eval('(u_t)',terms)
    return M, u_t

def check_dict(dictionary, inputs):
    input_vars = [sympy.Symbol(key) for key in inputs.keys()]
    if not 't' in [str(var) for var in input_vars]:
        raise ValueError("time must be marked with symbol 't'")
    func_var = sympy.Symbol('u')
    free_symbols = input_vars.copy()
    free_symbols += [sympy.Symbol('u_t'),]
    for d in dictionary:
        free_symbols += list(d.free_symbols)
    free_symbols = list(set(free_symbols))

    length_list = [len(str(item)) for item in free_symbols]
    indices = np.argsort(length_list)
    free_symbols = [free_symbols[i] for i in indices]
    length_list = [length_list[i] for i in indices]
    for i,symbol in enumerate(free_symbols):
        length = length_list[i]
        if length == 1:
            if not symbol in input_vars + [func_var,]:
                raise ValueError("Symbol '"+str(symbol) +"' illegal. Variable must be in ",input_vars + [func_var,] )
        elif length >= 3:
            if not  str(symbol)[0] == str(func_var):
                raise ValueError("Symbol '"+str(symbol) +"' illegal. The first character must be "+str(func_var))
            if not str(symbol)[1] == '_':
                raise ValueError("Symbol '"+str(symbol) +"' illegal. The second character must be '_'")
            for s in str(symbol)[2:]:
                if s not in [str(var) for var in input_vars]:
                    raise ValueError("Symbol '"+str(symbol) +"' illegal. Variable '"+ s + "' not in ",input_vars)
        else:
            raise ValueError("Symbol '"+str(symbol) +"' illegal. two-characters symbol not allowed")
    return free_symbols


def learning(inputs, u, dictionary, scalers=None,
             n_epochs=10000, batch_size=1000,
             alpha_pde_start=1, alpha_pde_end=50, alpha_l1=1e-5,
             warmup_nsteps=500,linearRegInterval=20,
             linearRegression=True,
             width=50,layers=4,
             lr=0.002,
             update_coef_in_dl=False,
             logfile=None):

    sorted_free_symbols = check_dict(dictionary,inputs)
    input_keys = inputs.keys()
    n_params = len(dictionary)
    dictionary = str(tuple(dictionary))
    in_size = len(inputs)
    n_points = len(u)
    x = sympy.symbols([key for key in inputs.keys()])

    if not logfile is None:
        outf = open(logfile,'w')
        #outf.write('%6s, %10s, %10s, %10s, %10s, '% ('Epoch','Loss_u', 'Loss_pde','Loss_l1','Loss_tot'))
        outf.write('Epoch,Loss_u,Loss_pde,Loss_l1,Loss_tot,')
        for i in range(n_params):
            #outf.write('%10s, '% ('p'+str(i)))
            if i == n_params-1:
                outf.write('p'+str(i))
            else:
                outf.write('p'+str(i)+',')
        outf.write('\n') 

    model = Model(in_size,width,layers,1,scalers).to(device)
    
    if update_coef_in_dl is True:
        comb = nn.Parameter(torch.randn(n_params,requires_grad=True,device=device,dtype=dtype))
        params = [{'params': model.parameters(), 'lr': lr},
                  {'params': comb, 'lr': .02}]
    else:
        comb = torch.randn(n_params,device=device,dtype=dtype)
        params = [{'params': model.parameters(), 'lr': lr}]

    optimizer = torch.optim.Adam(params)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,.9999)
    if linearRegression is True:
        lasso = Lasso(fit_intercept=False,alpha=alpha_l1)
    loss_u_item = [0.]
    loss_pde_item = [0.]
    loss_l1_item = [0.]
    loss_tot_item = [0.]
    for epoch in range(n_epochs):
        permutation = torch.randperm(n_points)
        for i in range(0,n_points, batch_size):
            def closure():
                optimizer.zero_grad()
                indices = permutation[i:i+batch_size]
                inputs_batch = {key:inputs[key][indices] for key in inputs.keys()}
                u_batch = u[indices]
                u_pred = model(torch.cat(list(inputs_batch.values()),dim=1))
                loss_u = torch.nn.functional.mse_loss(u_pred,u_batch)
                M, u_t= getdictionary(model, inputs_batch, dictionary, sorted_free_symbols)
                rhs = torch.sum(M*comb,dim=1,keepdims=True)
                loss_pde = torch.nn.functional.mse_loss(rhs, u_t)
                loss_l1 = torch.norm(comb,p=1)
                alpha_pde = alpha_pde_start + (alpha_pde_end-alpha_pde_start)/n_epochs*epoch
                if epoch >= warmup_nsteps:
                    loss = loss_u + alpha_pde*loss_pde + alpha_l1*loss_l1
                else:
                    loss = loss_u
                loss.backward(retain_graph=False)
                loss_u_item[0] = loss_u.item()
                loss_pde_item[0] = loss_pde.item()
                loss_l1_item[0] = loss_l1.item()
                loss_tot_item[0] = loss.item()
                return loss
            optimizer.step(closure)
        if linearRegression is True:
            if (epoch >= warmup_nsteps) and (epoch-warmup_nsteps)%linearRegInterval ==0:
                M, u_t= getdictionary(model, inputs, dictionary, sorted_free_symbols)
                lasso.fit(M.cpu().detach().numpy(),u_t.cpu().detach().numpy())
                comb.data = torch.tensor(lasso.coef_, device=device,dtype=dtype)
        
        if epoch % 50 ==0:
            print('Epoch: %4d,  Loss u: %.3e, Loss pde: %.3e, Loss_norm: %.3e, Loss tot: %.3e' % (epoch,loss_u_item[0],
                                                                     loss_pde_item[0],loss_l1_item[0],loss_tot_item[0]))
            coef = comb.data.tolist()
            format_w = len(coef)*" %- .4e  "
            print ('  coefs:'+format_w % tuple(coef))
        if not logfile is None:
            outf.write('%6d,%10.4e,%10.4e,%10.4e,%10.4e, '% (epoch,loss_u_item[0], loss_pde_item[0],loss_l1_item[0],loss_tot_item[0]))
            for i,para in enumerate(comb.data.cpu().detach().numpy()):
                if i == n_params-1:
                    outf.write('%10.4e' % (para))
                else:
                    outf.write('%10.4e,' % (para))
            outf.write('\n') 

        scheduler.step()
    if not logfile is None:
        outf.close()
    return  model
