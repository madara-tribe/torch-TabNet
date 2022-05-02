import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer



class UnsupervisedTBNet:
    def __init__(self, X_train, X_valid):
        self.X_train = X_train
        self.X_valid = X_valid

        self.max_epochs = 1000 if not os.getenv("CI", False) else 2
        # TabNetPretrainer
        self.unsupervised_model = TabNetPretrainer(
            #cat_idxs=cat_idxs,
            #cat_dims=cat_dims,
            cat_emb_dim=3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            mask_type='entmax', # "sparsemax",
            n_shared_decoder=1, # nb shared glu for decoding
            n_indep_decoder=1, # nb independent glu for decoding
        )
    
    def unsupervised_train(self):
        self.unsupervised_model.fit(
                X_train=self.X_train,
                eval_set=[self.X_valid],
                max_epochs=self.max_epochs, patience=5,
                batch_size=2048, virtual_batch_size=128,
                num_workers=0,
                drop_last=False,
                pretraining_ratio=0.8) 
        
    def predict(self):
        # Make reconstruction from a dataset
        reconstructed_X, embedded_X = self.unsupervised_model.predict(self.X_valid)
        assert(reconstructed_X.shape==embedded_X.shape)

        unsupervised_explain_matrix, unsupervised_masks = self.unsupervised_model.explain(self.X_valid)
        fig, axs = plt.subplots(1, 3, figsize=(20,20))

        for i in range(3):
            axs[i].imshow(unsupervised_masks[i][:50])
            axs[i].set_title(f"mask {i}")

        return self.unsupervised_model
    

class supervisedTBNet:
    def __init__(self, train_df, X_train, X_test, y_train, y_test, 
                    target_name, loaded_pretrain, epochs=3):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.loaded_pretrain = loaded_pretrain
        self.epochs = epochs
        self.local_interpretability_step = 3
        self.tabnet_params = dict(n_d=15, n_a=15,
                                n_steps=8,
                                gamma=0.2,
                                seed=10,
                                lambda_sparse=1e-3,
                                optimizer_fn=torch.optim.Adam,
                                optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                                mask_type="entmax",
                                scheduler_params=dict(
                                    max_lr=0.05,
                                    steps_per_epoch=int(self.X_train.shape[0] / 100),
                                    epochs=self.epochs,
                                    is_batch_level=True,
                                ),
                                verbose=5,
                            )
        self.src_df = train_df
    def train(self):
        # model
        self.model = TabNetRegressor(**self.tabnet_params)
        print('total epoch', self.epochs)
        self.model.fit(
            X_train=self.X_train, y_train=self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            eval_metric=['rmsle', 'mae', 'rmse', 'mse'],
            max_epochs=self.epochs,
            patience=30,
            batch_size=256,
            virtual_batch_size=128,
            num_workers=2,
            drop_last=False,
            loss_fn=torch.nn.functional.l1_loss,
            from_unsupervised=self.loaded_pretrain)
        
        
        print('show feature importance')
        self.plot_metric()
        #self.feature_importances()
        self.local_interpretability(self.X_test) 
        print('prediction')
        pred = self.model.predict(self.X_test)

        return self.model, pred, self.y_test
            
    def plot_metric(self):
        for param in ['loss', 'lr', 'val_0_rmsle', 'val_0_mae', 'val_0_rmse', 'val_0_mse']:
            plt.plot(self.model.history[param], label=param)
            plt.xlabel('epoch')
            plt.grid()
            plt.legend()
            plt.show()

    def local_interpretability(self, X_test):
        """どの特徴量を使うか decision making するのに用いた mask(Local interpretability)"""
        n_steps = self.local_interpretability_step
        explain_matrix, masks = self.model.explain(X_test)
        fig, axs = plt.subplots(n_steps, 1, figsize=(21, 3*n_steps))

        for i in range(n_steps):
            axs[i].imshow(masks[i][:50].T)
            axs[i].set_title(f"mask {i}")
            axs[i].set_yticks(range(len(self.src_df.columns[:-1])))
            axs[i].set_yticklabels(list(self.src_df.columns[:-1]))