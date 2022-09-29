import math
import pandas as pd
import numpy as np
import time
from datetime import datetime
import copy
import os
import random
from typing import Optional, Any, Union, Callable, Tuple
import mlflow

import torch
from torch import nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F


import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar

from kw_transformer import TransAm

from kw_multi_head_attention_forward import multi_head_attention_forward
from kw_transformer_layers import PositionalEncoding
from kw_transformer_multihead_attention import MultiheadAttention
from kw_TransformerEncoderLayer import TransformerEncoderLayer
from kw_transformer_functions import calculate_metrics, RMSELoss, RMSPELoss, plot_dataset, inverse_transform, format_predictions, plot_predictions,final_split,final_dataload

df=pd.read_csv("dataset.csv")
df = df.rename(columns={'vol_future': 'value'})
df['date'] = pd.to_datetime(df['date'])
df = df.set_index(['date'])
df.index = pd.to_datetime(df.index)





@hydra.main(version_base='1.2',config_path="config/model", config_name="experiment")
def main(cfg):  


    X_train, X_test, y_train, y_test = final_split(df, 'value', 0.15)

    train_loader,test_loader, test_loader_one,scaler=final_dataload(cfg.params.batch_size,X_train,X_test, y_train, y_test)

    feature_size = len(X_train.columns) #input_dim 

    loss_fn = RMSELoss

    # train
    model = TransAm(loss_fn,cfg.params.batch_size,feature_size,cfg.params.num_layers,
                    cfg.params.dropout,cfg.params.nhead,cfg.params.attn_type,
                    cfg.params.lr,cfg.params.weight_decay)


    early_stop_callback = EarlyStopping(monitor="val_loss", patience=cfg.params.patience, verbose=0, 
                                        mode="min")

    
    checkpoint_callback = ModelCheckpoint(dirpath="modelcheckpoint/"+cfg.model_checkpoint.outputdir,filename='{epoch}-{val_loss:.3f}'
                                      ,save_top_k=1, monitor="val_loss")
  
 
    
    #lr_monitor = LearningRateMonitor(logging_interval='step')


    hyperparameters = dict(num_layers=cfg.params.num_layers,feature_size=feature_size ,
                          batch_size=cfg.params.batch_size,dropout=cfg.params.dropout,
                           nhead=cfg.params.nhead,attn_type=cfg.params.attn_type,learning_rate=cfg.params.lr,
                           weight_decay=cfg.params.weight_decay,n_epochs=cfg.params.n_epochs,loss_fn=loss_fn.__name__)

    mlflow.pytorch.autolog()


    #logger = MLFlowLogger(save_dir="./loggercheckpoint",run_name="hydra_mlflow")
    #logger = TensorBoardLogger(save_dir="./loggercheckpoint", version=1, name='kw_tensorboardlogger')
    #logger = TensorBoardLogger(save_dir=".",version=3, name='loggercheckpoint')


    trainer = pl.Trainer(callbacks=[TQDMProgressBar(refresh_rate=0),early_stop_callback,checkpoint_callback], 
                         max_epochs=cfg.params.n_epochs,logger=False,
                         gpus=1 if torch.cuda.is_available() else None)



    with mlflow.start_run(experiment_id=cfg.mlflow.experiment_id,run_name = cfg.mlflow.run_name) as run:
        mlflow.log_params(hyperparameters)
        trainer.fit(model, train_loader, test_loader)

    
if __name__ == "__main__":
    main()    