#import pytorch_lightning as pl
import torch
from torch.nn import functional as F


class TransformerModel(torch.nn.Module):
    def __init__(self, _transformer, _tokenizer, opt='adam', lr=5e-5, decay=2e-5, minlr=1e-5, lr_strategy="simple", max_len=50):
        super(TransformerModel, self).__init__()
        self.lr = lr
        self.decay = decay
        self.opt = opt
        self.minlr = minlr
        self.lrStrategy=lr_strategy
        # not the best model...
        dim= lambda x: x['dim'] if "dim" in x else (x['hidden_size'] if "hidden_size" in x else 768)
        self.encoder = _transformer
        self.tokenizer = _tokenizer
        #print(_transformer.config.to_dict())
        hiddensize = dim(_transformer.config.to_dict())
        self.dense1 = torch.nn.Sequential(torch.nn.Linear(in_features=hiddensize, out_features=64), torch.nn.LeakyReLU())
        self.dense2 = torch.nn.Linear(in_features=64, out_features=32)
        self.drop = torch.nn.Dropout(p=0.25)
        self.classifier = torch.nn.Linear(in_features=32, out_features=2)

        self.dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.max_length = max_len
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.to(device=self.dev)
        self.best_measure = None
        self.best_model_name = None

    def forward(self, x):        
        x = self.tokenizer(x, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.dev)
        x = self.encoder(**x)[0][:, 0]
        x = self.dense1(x)
        x = torch.nn.functional.relu(self.dense2(x))
        x = self.classifier(x)

        return x

    def predict_step(self, batch):
        # OPTIONAL
        #with torch.no_grad():
            x = batch
            y_hat = self.forward(x)
            preds = torch.max(y_hat, 1).indices
            del x
            del y_hat
            return preds
  

    def configure_optimizers(self):

        print("Configure Optimizers")
        if self.lrStrategy=="dynamic":
            return self.configure_optimizers_dynamic1()
        return  self.configure_optimizers_simple()


    def configure_optimizers_simple(self):

        print("Configured Simple {} with lr: {}".format(self.opt, self.lr ))
        if self.opt=='adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.decay)
        elif self.opt=='rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.decay)


    def configure_optimizers_dynamic(self, multiplier=1, increase=0.1):

        print("Configured Dynamic {} with starting lr: {} and X-increase: {}".format(self.opt, self.lr, increase))
        params = []
        for l in self.encoder.encoder.layer:

            params.append({'params':l.parameters(), 'lr':self.lr*multiplier}) 
            multiplier += increase

        try:
            params.append({'params':self.encoder.pooler.parameters(), 'lr':self.lr*multiplier})
        except:
            print('Warning: No Pooler layer found')

        params.append({'params': self.dense1.parameters()})
        params.append({'params': self.dense2.parameters()})
        params.append({'params': self.classifier.parameters()})

        if self.opt == 'adam':
            return torch.optim.Adam(params, lr=self.lr*multiplier, weight_decay=self.decay)
        elif self.opt == 'rmsprop':
            return torch.optim.RMSprop(params, lr=self.lr*multiplier, weight_decay=self.decay)
        
        
    def configure_optimizers_dynamic1(self):
        print("Configured Dynamic {} with starting lr: {} and Exponential Increase".format(self.opt, self.lr))
        params = []
        i=0
        layernum=12
        diff =abs(self.lr -self.minlr)
        for l in self.encoder.encoder.layer:

            params.append({'params':l.parameters(), 'lr': self.minlr+diff**(layernum)}) 
            layernum-=1

        try:
            params.append({'params':self.encoder.pooler.parameters(), 'lr':self.lr})
        except:
            print('Warning: No Pooler layer found')

        params.append({'params': self.dense1.parameters()})
        params.append({'params': self.dense2.parameters()})
        params.append({'params': self.classifier.parameters()})

        if self.opt == 'adam':
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.decay)
        elif self.opt == 'rmsprop':
            return torch.optim.RMSprop(params, lr=self.lr, weight_decay=self.decay)

            
    
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.dev))

    def save(self, path):
        torch.save(self.state_dict(), path)
# --------------------------------------------------------------------------
