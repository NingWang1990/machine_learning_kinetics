
# author Ning Wang, Aug 2020
# to do
# clean up device

import sympy
import torch
#torch.manual_seed(12)
#torch.cuda.manual_seed(12)
torch.manual_seed(1001)
torch.cuda.manual_seed(1001)
torch.cuda.manual_seed_all

dtype = torch.float
#device = torch.device("cpu")
import torch.nn as nn
from torch.autograd import grad
from torch.autograd import Variable
import numpy as np
from sklearn.linear_model import Lasso
device = torch.device("cuda:0")


class modified_tanh(nn.Module):
    def __init__(self):
        super(modified_tanh,self).__init__()

    def forward(self,x):
        return 0.5*(1-torch.tanh(x))

class modified_shifting_tanh(nn.Module):
    def __init__(self,device=device):
        super(modified_shifting_tanh,self).__init__()
        a_init = np.ones((1,size))
        b_init = np.zeros((1,size))
        self.a = nn.Parameter(torch.tensor(a_init,requires_grad=True,device=device,dtype=dtype))
        self.b = nn.Parameter(torch.tensor(b_init,requires_grad=True,device=device,dtype=dtype))
    def forward(self,x):
        return 0.5*(1-torch.tanh(self.a*(x-self.b)))

class shifting_tanh(nn.Module):
    def __init__(self,size=1,device=device):
        super(shifting_tanh,self).__init__()
        a_init = np.ones((1,size))
        b_init = np.zeros((1,size))
        self.a = nn.Parameter(torch.tensor(a_init,requires_grad=True,device=device,dtype=dtype))
        self.b = nn.Parameter(torch.tensor(b_init,requires_grad=True,device=device,dtype=dtype))
    def forward(self,x):
        return torch.tanh(self.a*(x-self.b))


class sine_activation(nn.Module):
    def __init__(self):
        super(sine_activation,self).__init__()

    def forward(self,x):
        return torch.sin(x)

class ResNetBlock(nn.Module):
    def __init__(self, hidden_size,activation):
        super(ResNetBlock, self).__init__()
        self.linear_layer = nn.Linear(hidden_size,hidden_size,bias=True)
        self.activation = activation
    def forward(self,x):
        return self.activation(self.linear_layer(x)) + x

def get_activation(activation,device=device,size=None):
    if activation == 'tanh':
        activation = nn.Tanh()
    elif activation =='sigmoid':
        activation = nn.Sigmoid()
    elif activation == 'modified_tanh':
        activation = modified_tanh()
    elif activation == 'shifting_tanh':
        activation = shifting_tanh(size=size)
    elif activation == 'modified_shifting_tanh':
        activation=modified_shifting_tanh(device=device)
    elif activation == 'sine':
        activation = sine_activation()
    else:
        raise NotImplementedError()
    return activation

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden_layers, out_size, 
                 scalers=None,batch_normalization=False,resnet=False,dropout_prob=0,
                 activation='tanh', device = device):
        super(Model, self).__init__()
        #if scalers == None:
        #    self.scalers = torch.ones((1,input_size),device=device, dtype=dtype)
        #else:
        #    self.scalers = scalers
        #self.scalers = 1.
        layers = [nn.Linear(input_size,hidden_size,bias=True),get_activation(activation,device,size=hidden_size),]
        layers.append(nn.Dropout(dropout_prob))
        if batch_normalization == True:
            layers += [nn.BatchNorm1d(hidden_size),]
        for i in range(n_hidden_layers):
            if resnet is False:
                layers += [nn.Linear(hidden_size,hidden_size,bias=True), get_activation(activation,device,size=hidden_size),]
            else:
                layers += [ResNetBlock(hidden_size, get_activation(activation,device,size=hidden_size)),]
            if batch_normalization == True:
                layers += [nn.BatchNorm1d(hidden_size),]
            layers.append(nn.Dropout(dropout_prob))
        layers += [nn.Linear(hidden_size, out_size,bias=True),]
        self.layers = nn.Sequential(*layers) 
    def forward(self,x):
        return self.layers(x)

def get_M_ut(model,inputs,dictionary,sorted_free_symbols):
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

def laplace_operator(f,inputs):
    if ('x' in inputs.keys()) and ('y' in inputs.keys()):
        dim = 2
    elif ('x' in inputs.keys()):
        dim = 1
    else:
        raise RuntimeError('spatial variables not found ')
    f_x = grad(f,inputs['x'],create_graph=True,
                     grad_outputs=torch.ones_like(f))[0]
    f_xx = grad(f_x,inputs['x'],create_graph=True,
                     grad_outputs=torch.ones_like(f_x))[0]
    if dim == 2:
        f_y = grad(f,inputs['y'],create_graph=True,
                     grad_outputs=torch.ones_like(f))[0]
        f_yy = grad(f_y,inputs['y'],create_graph=True,
                     grad_outputs=torch.ones_like(f_y))[0]
    if dim == 1:
        return f_xx
    elif dim == 2:
        return f_xx + f_yy

def get_losses(in_dict,u_true,model,coefs,dictionary,sorted_free_symbols,
             fit_intercept,laplace_rhs,loss_type, intercept=None):
    u_pred = model(torch.cat(list(in_dict.values()),dim=1).to(device))
    M, u_t = get_M_ut(model, in_dict, dictionary, sorted_free_symbols)
    #rhs = torch.sum(M*coefs,dim=1,keepdims=True) 
    n_params = len(coefs)
    rhs = M[:,0:1]*coefs[0]
    for k in range(1,n_params):
        rhs += M[:,k:(k+1)]*coefs[k]
    if fit_intercept is True:
        rhs = rhs + intercept
    if laplace_rhs is True:
        rhs = laplace_operator(rhs, in_dict)
    if loss_type == 'mse':
        loss_u = torch.nn.functional.mse_loss(u_pred,u_true)
        loss_pde = torch.nn.functional.mse_loss(rhs, u_t)
    elif loss_type == 'rmse':
        loss_u = torch.sqrt(torch.nn.functional.mse_loss(u_pred,u_true))
        loss_pde = torch.sqrt(torch.nn.functional.mse_loss(rhs, u_t))
    elif loss_type == 'mae':
        loss_u = torch.mean(torch.abs(u_pred-u_true))
        loss_pde = torch.mean(torch.abs(rhs-u_t))
    loss_para = torch.norm(torch.tensor(coefs,device=device,dtype=dtype),p=1)
    return loss_u, loss_pde, loss_para

initializing_method_implemented = ['zero','random']
loss_type_implemented = ['mse','rmse','mae']
def learning(in_train, u_train, in_valid, u_valid, dictionary, scalers=None,
             n_epochs=10000, batch_size=1000,
             batch_valid=1000,
             alpha_pde=1., alpha_l1=1e-5,
             linearRegStart=700,linearRegInterval=200,
             batch_normalization=False,
             linearRegression=True,
             resnet=False,
             dropout_prob=0,
             width=50,layers=4,
             warmup_nsteps=0,
             lr_u=0.002,
             lr_coefs=0.002,
             lr_multi_factor=1.,
             fixed_coefs={},
             fit_intercept=False,
             initializing_method='zero',
             activation='tanh',
             device = device,
             log_batch=False,
             loss_type='mse',
             laplace_rhs=False,
             save_best_in_batch_loop=True,
             best_model_filename='best_model.pt',
             best_coefs_filename='best_coefs.txt',
             logfile=None):

    """
    fixed_coefs.............dict,example:{0:0.11, 1:-0.1} 
    lr_coefs................float or list. if float, same learning rate for all coefs
                                           if list, use a separate learning rate for each coef
    """
    if not initializing_method in initializing_method_implemented:
        raise ValueError('initializing mehod must be in ',initializing_method_implemented)
    if not loss_type in loss_type_implemented:
        raise ValueError('loss_type must be in ',loss_type_implemented)
    sorted_free_symbols = check_dict(dictionary,in_train)
    input_keys = in_train.keys()
    n_params = len(dictionary)
    if not isinstance(lr_coefs, float):
        if not len(lr_coefs) == n_params:
            raise ValueError('length of lr_coefs should be equal to length of dictionary')
    else:
        lr_coefs = n_params *[lr_coefs,]
    dictionary = str(tuple(dictionary))
    in_size = len(in_train)
    n_points = len(u_train)
    x = sympy.symbols([key for key in in_train.keys()])

    if not logfile is None:
        outf = open(logfile,'w')
        #outf.write('%6s, %10s, %10s, %10s, %10s, '% ('Epoch','Loss_u', 'Loss_pde','loss_para','Loss_tot'))
        if log_batch is True:
            outf.write('Batch,')
        else:
            outf.write('Epoch,')
        outf.write('LossTrain_u,LossTrain_pde,lossTrain_para,LossTrain_tot,')
        outf.write('LossValid_u,LossValid_pde,lossValid_para,LossValid_tot,')
        if fit_intercept is True:
            outf.write('intercept,')
        for i in range(n_params):
            #outf.write('%10s, '% ('p'+str(i)))
            if i == n_params-1:
                outf.write('coef'+str(i))
            else:
                outf.write('coef'+str(i)+',')
        outf.write('\n') 
        outf.close()
    model = Model(in_size,width,layers,1,scalers,batch_normalization,resnet,dropout_prob,activation,device)
    model = nn.DataParallel(model)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters in neural nets: %d' % (total_params,))

    # initial kappa with a positive value
    coefs = [nn.Parameter(torch.tensor(0.,requires_grad=True,device=device,dtype=dtype))] 
    if initializing_method == 'random':
        randn_values = torch.randn(n_params-1)
        coefs += [nn.Parameter(torch.tensor(val,requires_grad=True,device=device,dtype=dtype)) for val in randn_values]
    elif initializing_method == 'zero':
        coefs += [nn.Parameter(torch.tensor(0.,requires_grad=True,device=device,dtype=dtype)) for i in range(n_params-1)]
    if fit_intercept is True:
        intercept = nn.Parameter(torch.tensor(0.,requires_grad=True,device=device,dtype=dtype))
    else:
        intercept = None
    for key in fixed_coefs.keys():
        coefs[key].data = torch.tensor(fixed_coefs[key],device=device,dtype=dtype)
    params = [{'params': model.parameters(), 'lr': lr_u}]
    for i in range(n_params):
        params += [{'params': coefs[i], 'lr': lr_coefs[i]}]
    if fit_intercept is True:
        params += [{'params': intercept, 'lr': lr_u}]
    optimizer = torch.optim.Adam(params)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,lr_multi_factor)
    if linearRegression is True:
        lasso = Lasso(fit_intercept=fit_intercept,alpha=alpha_l1)
    batch_count = 0
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.00)
    model.apply(init_weights)
    lowest_loss = np.inf
   
    if (log_batch is True) or ( save_best_in_batch_loop is True):
        calculate_valid_loss_in_batch = True
    else:
        calculate_valid_loss_in_batch = False

    for epoch in range(n_epochs):
        permutation = torch.randperm(n_points)
        if (log_batch is True) and (not logfile is None):
            outf = open(logfile,'a')
        for i in range(0,n_points, batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i+batch_size]
            inputs_batch = {key:in_train[key][indices] for key in in_train.keys()}
            u_batch = u_train[indices]
            loss_u, loss_pde, loss_para = get_losses(inputs_batch,u_batch,model,coefs,dictionary,sorted_free_symbols,
                                                        fit_intercept,laplace_rhs,loss_type, intercept)
            if epoch >= warmup_nsteps:
                loss = loss_u + alpha_pde*loss_pde + alpha_l1*loss_para
            else:
                loss = loss_u
            loss_train_u = loss_u.item()
            loss_train_pde = loss_pde.item()
            loss_train_para = loss_para.item()
            loss_train_tot = loss.item()
            loss.backward(retain_graph=False)
            optimizer.step()
            # clamp kappa to positive range
            #coefs[0].data = torch.clamp(coefs[0].data,0., 1e30)
            # fix selected coefficients
            for key in fixed_coefs.keys():
                coefs[key].data = torch.tensor(fixed_coefs[key],device=device,dtype=dtype) 
            
            if calculate_valid_loss_in_batch is True:
                loss_valid_u = 0.
                loss_valid_pde = 0.
                loss_valid_para = 0.
                for j in range(0,len(u_valid),batch_valid):
                    inputs_batch =  {key:in_valid[key][j:(j+batch_valid)] for key in in_valid.keys()}
                    u_batch = u_valid[j:(j+batch_valid)]
                    loss_u, loss_pde, loss_para = get_losses(inputs_batch,u_batch,model,coefs,dictionary,sorted_free_symbols,
                                                               fit_intercept,laplace_rhs,loss_type, intercept)
                    loss_valid_u += loss_u.item()*len(u_batch)
                    loss_valid_pde += loss_pde.item()*len(u_batch)
                    loss_valid_para += loss_para.item()*len(u_batch)
                loss_valid_u /= len(u_valid)
                loss_valid_pde /= len(u_valid)
                loss_valid_para /= len(u_valid)
                loss_valid_tot = loss_valid_u + alpha_pde*loss_valid_pde + alpha_l1*loss_valid_para
            
            if (save_best_in_batch_loop is True) and ( loss_valid_tot < lowest_loss):
                torch.save(model.state_dict(), best_model_filename) 
                lowest_loss = loss_valid_tot
                outc = open(best_coefs_filename,'w')
                if fit_intercept is True:
                    outc.write('intercept,')
                for i in range(n_params):
                    if i == n_params-1:
                        outc.write('coef'+str(i))
                    else:
                        outc.write('coef'+str(i)+',')
                outc.write('\n')
                if fit_intercept is True:
                    outc.write('%10.4e,' %(float(intercept.tolist()),))
                for i,para in enumerate(torch.tensor(coefs).tolist()):
                    if i == n_params-1:
                        outc.write('%10.4e' % (para))
                    else:
                        outc.write('%10.4e,' % (para))
                outc.write('\n') 
                outc.close()

            if (log_batch is True) and (not logfile is None):
                outf.write('%6d,%10.4e,%10.4e,%10.4e,%10.4e, '% (batch_count,
                            loss_train_u, loss_train_pde,loss_train_para,loss_train_tot))
                outf.write('%10.4e,%10.4e,%10.4e,%10.4e, '% (loss_valid_u, 
                            loss_valid_pde,loss_valid_para,loss_valid_tot))
                if fit_intercept is True:
                    outf.write('%10.4e,' %(float(intercept.tolist()),))
                for i,para in enumerate(torch.tensor(coefs).tolist()):
                    if i == n_params-1:
                        outf.write('%10.4e' % (para))
                    else:
                        outf.write('%10.4e,' % (para))
                outf.write('\n') 
            batch_count += 1

        if (log_batch is True) and (not logfile is None):
            outf.close()
       
        if calculate_valid_loss_in_batch is False:
            loss_valid_u = 0.
            loss_valid_pde = 0.
            loss_valid_para = 0.
            for j in range(0,len(u_valid),batch_valid):
                inputs_batch =  {key:in_valid[key][j:(j+batch_valid)] for key in in_valid.keys()}
                u_batch = u_valid[j:(j+batch_valid)]
                loss_u, loss_pde, loss_para = get_losses(inputs_batch,u_batch,model,coefs,dictionary,sorted_free_symbols,
                                                               fit_intercept,laplace_rhs,loss_type, intercept)
                loss_valid_u += loss_u.item()*len(u_batch)
                loss_valid_pde += loss_pde.item()*len(u_batch)
                loss_valid_para += loss_para.item()*len(u_batch)
            loss_valid_u /= len(u_valid)
            loss_valid_pde /= len(u_valid)
            loss_valid_para /= len(u_valid)
            loss_valid_tot = loss_valid_u + alpha_pde*loss_valid_pde + alpha_l1*loss_valid_para

        if (save_best_in_batch_loop is False) and ( loss_valid_tot < lowest_loss):
            torch.save(model.state_dict(), best_model_filename) 
            lowest_loss = loss_valid_tot
            outc = open(best_coefs_filename,'w')
            if fit_intercept is True:
                outc.write('intercept,')
            for i in range(n_params):
                if i == n_params-1:
                    outc.write('coef'+str(i))
                else:
                    outc.write('coef'+str(i)+',')
            outc.write('\n')
            if fit_intercept is True:
                outc.write('%10.4e,' %(float(intercept.tolist()),))
            for i,para in enumerate(torch.tensor(coefs).tolist()):
                if i == n_params-1:
                    outc.write('%10.4e' % (para))
                else:
                    outc.write('%10.4e,' % (para))
            outc.write('\n') 
            outc.close()

        if linearRegression is True:
            if (epoch >= linearRegStart) and (epoch-linearRegStart)%linearRegInterval ==0:
                M, u_t= get_M_ut(model, in_train, dictionary, sorted_free_symbols)
                lasso.fit(M.cpu().detach().numpy(),u_t.cpu().detach().numpy())
                for i in range(len(coefs)):
                    coefs[i].data = torch.tensor(lasso.coef_[i], device=device,dtype=dtype)
                if fit_intercept is True:
                    intercept.data = torch.tensor(lasso.intercept_[0],device=device,dtype=dtype)
        

        if epoch % 50 ==0:
            print('Epoch: %4d,  LossTr u: %.3e, LossTr pde: %.3e, lossTr para: %.3e, LossTr tot: %.3e' % (epoch,loss_train_u,
                                                                loss_train_pde,loss_train_para,loss_train_tot))
            coefs_list = torch.tensor(coefs).tolist()
            format_w = len(coefs_list)*" %- .4e  "
            print ('  coefs:'+format_w % tuple(coefs_list))
            if fit_intercept is True:
                print ('  intercept: %- .4e' % (float(intercept.tolist()),))
        
        if (not logfile is None) and (log_batch is False):
            outf = open(logfile,'a')
            outf.write('%6d,%10.4e,%10.4e,%10.4e,%10.4e, '% (epoch,loss_train_u, loss_train_pde,loss_train_para,loss_train_tot))
            outf.write('%10.4e,%10.4e,%10.4e,%10.4e, '% (loss_valid_u, 
                            loss_valid_pde,loss_valid_para,loss_valid_tot))
            if fit_intercept is True:
                outf.write('%10.4e,' %(float(intercept.tolist()),))
            for i,para in enumerate(torch.tensor(coefs).tolist()):
                if i == n_params-1:
                    outf.write('%10.4e' % (para))
                else:
                    outf.write('%10.4e,' % (para))
            outf.write('\n') 
            outf.close()
        scheduler.step()
    return  model
