import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from kw_transformer_layers import PositionalEncoding
import torch
from torch import nn

class LSTMModel(pl.LightningModule):
    def __init__(self, loss_fn,batch_size=32,input_dim=1, hidden_dim=64, layer_dim=3, output_dim=1, dropout=0.1,learning_rate=1e-5,weight_decay=1e-6):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.loss_fn=loss_fn
        self.input_dim=input_dim
        self.batch_size=batch_size
        self.hidden_dim = hidden_dim #lookback period
        self.layer_dim = layer_dim
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.loss_fn = loss_fn
        self.dropout=dropout
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=self.dropout).cuda()

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim).cuda()
        self.save_hyperparameters()
    def forward(self, x):
        # Initializing hidden state for first input with zeros
        #h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        #c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        #out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().cuda()
         
    
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
      
        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        
        return out




    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    
    
    def train_dataloader(self):
        # REQUIRED
        # This is an essential function. Needs to be included in the code
               
        return DataLoader(self.train_set,batch_size=128,num_workers=32)
        
    def val_dataloader(self):
        # OPTIONAL
        #loading validation dataset
        return DataLoader(self.val_set, batch_size=128,num_workers=32)

    def test_dataloader(self):
        # OPTIONAL
        # loading test dataset
        return DataLoader(MNIST(os.getcwd(), train=False, download=False, transform=transforms.ToTensor()), batch_size=128,num_workers=32)
    
    #def RMSE_loss(self, logits, labels):
    #    return self.loss_fn(logits, labels)
    
    #def on_train_start(self):
    #    self.logger.log_hyperparams({"hp/learning_rate": self.learning_rate, 
    #                                           "hp/batch_size": self.batch_size})
    #    kw_dict=dict()
        
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view([self.batch_size, -1, self.input_dim]) 
        pred = self.forward(x)
        # The actual forward pass is made on the 
        #input to get the outcome pred from the model
        pred = pred.view(-1,1)
        loss = self.loss_fn(pred, y)
     
        self.log('training_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view([self.batch_size, -1, self.input_dim])
        pred = self.forward(x)
        pred = pred.view(-1,1)
        loss = self.loss_fn(pred, y)
        self.log('val_loss', loss)
        #print('val_loss:',loss)
        return loss
        
    def test_step(self, test_batch, batch_idx,batch_size=1):
        x, y = test_batch
        x = x.view([batch_size, -1, self.input_dim])
  
        pred = self.forward(x)
        pred = pred.view(-1,1)

        loss = self.loss_fn(pred, y)
        self.log('Test loss', loss)
     
        return loss

        
    #    print(len(losses)) ## This will be same as number of validation batches
    def predict_step(self,batch,batch_idx):
        X_batch, Y_batch = batch
        X_batch=X_batch.unsqueeze(0)
        preds = self(X_batch.float())


        return preds
    
    