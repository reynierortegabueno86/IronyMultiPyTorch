import argparse
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import shuffle
import pandas as pd, numpy as np
import config as conf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorchirodatamodule import IronyDataModule, Tokenization, TextCompose
from transformermodels import TransformerModel
from utils import getModelTokenizerConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from sklearn.metrics import classification_report, accuracy_score, fbeta_score
import gc
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
SEED = 1234
seed_everything(SEED)



def check_params(args=None):
    parse = argparse.ArgumentParser(description='Method for automatically irony detection in multiple languages')
    parse.add_argument('-i', '--input', help='File with the input data', required=False, default="training.tsv")
    parse.add_argument('-o', '--output', help='File path for writing the results and metrics', required=False,
                       default="output.out")
    parse.add_argument('-d', '--dropout', help='Dropout values used to reduce the overfitting ', required=False,
                       type=float, default=conf.defaultDp)
    parse.add_argument('-ml', '--maxlength', help='Max length of the sequences used for training', required=False,
                       type=int, default=conf.defaultMlenght)
    parse.add_argument('-p', '--epochs', help='Number of epoch used in the training phase', required=False, type=int,
                       default=conf.defaultEpoch)
    parse.add_argument('-b', '--batchsize', help='Batch size used in the training process', required=False, type=int,
                       default=conf.defaulBatchsize)
    parse.add_argument('-z', '--optimizer', help='Method used for parameters optimizations', required=False, type=str,
                       default=conf.defaultOp)
    parse.add_argument('-r', '--learning', help='Value for the learning rate in the optimizer', required=False,
                       type=float, default=conf.defaultLr)
    parse.add_argument('-sr', '--lrstategy', help='Value for the learning rate in the optimizer', required=False,
                       type=str, choices=['dynamic', 'simple'], default=conf.defaulLrStrategy)
    parse.add_argument('-dr', '--lrdecay', help='Value for the weigth decay in the optimizer', required=False,
                       type=float, default=conf.defaultLr)
    parse.add_argument('-mr', '--minlearning', help='Value for the minimal learning rate in the optimizer',
                       required=False, type=float, default=conf.defaultLr)
    parse.add_argument('-md', '--model', help='Model', required=False,
                       choices=["mbert", "xlm-roberta", "beto", "spanbert", "roberta", "bert", 'bertweet', "alberto",
                                'umberto'],
                       default='bertweet')
    parse.add_argument('-k', '--folds', help='Number of partitions in the cross-validation methods', required=False,
                       type=int, default=conf.defaultFold)
    parse.add_argument('-l', '--language', help='Language of the model', required=False, default="en",
                       choices=['ml', 'es', 'en', 'it'])
    parse.add_argument('-w', '--patience', help='Patience for the early stopping criterion', required=False, type=int,
                       default=conf.defaultEpatience)

    results = parse.parse_args(args)
    return results


if __name__ == "__main__":
    import sys
    from utils import SqliteDB, replace_param_value
    parameters = check_params(sys.argv[1:])
    database="database/urls.sqlite3"
    sql3=SqliteDB(database)
    #fileoutname=replace_param_value(parameters._get_kwargs(), {'input':sql3.shorten, 'output':sql3.shorten})
    fileoutname=parameters._get_kwargs()
    fileoutname.sort()
    shorten="%%%".join([f'{x[0]}_{str(x[1])}' for x in fileoutname ])
    fileoutname=sql3.shorten(shorten)
    print(fileoutname)
    sql3.close()
    # collection="EnBarbieriEtAl14"
    # Load the pandas dataframe with the irony collection
    ironyData = pd.read_csv(parameters.input, sep="\t")
    # ironyData=pd.read_csv("data/EsIroSvA19/training.tsv", sep="\t")
    ironyData = shuffle(ironyData, random_state=SEED)
    ironyID, ironyLabels, ironyData = ironyData["id"], ironyData["irony"], ironyData[["irony", "preprotext"]]
    # ironyLabels=ironyData["irony"]
    # ironyData=ironyData[["irony","preprotext"]]
    lab = ironyLabels.to_numpy()
    # print(lab, sum(lab), '/', len(lab))
    # CREATE PARTITIONS
    skfold = StratifiedKFold(n_splits=parameters.folds, shuffle=True, random_state=SEED)
    overallACC, overallF1 = 0.0, 0.0
    # Loading the pretrained model
    # pretrainedmodel, tokenizer = getModelTokenizerConfig(parameters.model, parameters.language)

    history = []
    os.makedirs(parameters.output, exist_ok=True)
    with open(parameters.output+os.sep+fileoutname, 'w')  as myout:
        for fold, (train_ids, test_ids) in enumerate(skfold.split(X=ironyData.to_numpy(), y=ironyLabels)):

            history.append({'loss': [], 'acc': [], 'dev_loss': [], 'dev_acc': []})
            torch.cuda.empty_cache()
            print('----------------------------------------------------')
            print(f'******Evaluating the FOLD {fold + 1}/{parameters.folds}')
            print('----------------------------------------------------')
            pretrainedmodel, tokenizer = getModelTokenizerConfig(parameters.model, parameters.language)
            # Defining the data data pipeline
            data_transform = TextCompose(
                [Tokenization(tokenizer, paddingstrategy='max_length', maxlength=parameters.maxlength, truncate=True)])
            # Obtaining the data split for training and test.
            iroTrain, iroTest = ironyData.iloc[train_ids], ironyData.iloc[test_ids]
            print("#" * 50)
            print("TRAINING:", sum(lab[train_ids]), "/", len(lab[train_ids]))
            print("TEST ", sum(lab[test_ids]), "/", len(lab[test_ids]))
            print("#" * 50)
            data = IronyDataModule(train=iroTrain, test=iroTest,
                                   batch_size=parameters.batchsize)  # , transform=data_transform)
            # Callbacks
            # checkpoint = ModelCheckpoint(monitor='vacc', dirpath='/tmp/',
            #                              filename='sample-mnist-{epoch:02d}-{val_loss:.2f}')
            # early_stop = EarlyStopping(monitor='vacc', min_delta=0.00, patience=10, verbose=False, mode='max')
            # CREATE THE MODEL (Using Pytorch-Lightning)
            model = TransformerModel(pretrainedmodel, tokenizer, lr=parameters.learning, opt=parameters.optimizer,
                                     lr_strategy=parameters.lrstategy, minlr=parameters.minlearning)
            optimizer = model.configure_optimizers()

            data.setup("fit")
            trainloader = data.train_dataloader()
            devloader = data.val_dataloader()
            test = data.test_dataloader()
            del data
            
            patience=0
            batches = len(trainloader)
            for epoch in range(parameters.epochs):
                if patience >=parameters.patience: break
                
                running_loss = 0.0
                perc = 0
                acc = 0
                model.train()
                for j, data in enumerate(trainloader, 0):
                    torch.cuda.empty_cache()
                    inputs, labels = data['X'], data['y'].to(model.dev)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = model.loss_criterion(outputs, labels)
                    # print(outputs.shape, labels.shape)
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        if j == 0:
                            acc = ((torch.max(outputs, 1).indices == labels).sum() / len(labels)).cpu().numpy()
                            running_loss = loss.item()
                        else:
                            acc = (acc + ((torch.max(outputs, 1).indices == labels).sum() / len(
                                labels)).cpu().numpy()) / 2.0
                            running_loss = (running_loss + loss.item()) / 2.0

                    if (j + 1) * 100.0 / batches - perc >= 1 or j == batches - 1:
                        perc = (1 + j) * 100.0 / batches
                        print('\r Epoch:{} step {} of {}. {}% loss: {}'.format(epoch + 1, j + 1, batches,
                                                                               np.round(perc, decimals=1),
                                                                               np.round(running_loss, decimals=3)),
                              end="")

                model.eval()
                history[-1]['loss'].append(running_loss)
                with torch.no_grad():
                    out = None
                    log = None
                    for k, data in enumerate(devloader, 0):
                        torch.cuda.empty_cache()
                        inputs, label = data['X'], data['y'].to(model.dev)
                        dev_out = model(inputs)
                        if k == 0:
                            out = dev_out
                            log = label
                        else:
                            out = torch.cat((out, dev_out), 0)
                            log = torch.cat((log, label), 0)

                    dev_loss = model.loss_criterion(out, log).item()
                    dev_acc = ((torch.max(out, 1).indices == log).sum() / len(log)).cpu().numpy()
                    #print(torch.max(out, 1).indices.sum(), log.sum())
                    history[-1]['acc'].append(acc)
                    history[-1]['dev_loss'].append(dev_loss)
                    history[-1]['dev_acc'].append(dev_acc)

                if model.best_measure is None or model.best_measure < dev_acc:  # here you must set your saveweights criteroin
                    model.best_measure = dev_acc
                    model.best_model_name='models/bestmodel_split_{}.pt'.format(fold + 1)
                    model.save(model.best_model_name)
                    #model.save('bestmodelo_split_{}.pt'.format(fold + 1))
                    patience=0
                
                patience+=1

                print(" acc: {} ||| dev_loss: {} dev_acc: {}".format(np.round(acc, decimals=3),
                                                                     np.round(dev_loss, decimals=3),
                                                                     np.round(dev_acc, decimals=3)))

            # ---------------------------------------------------------------------------------------------------------------------------------------------
            model.load(model.best_model_name)
            model.eval()
            with torch.no_grad():
                predictions = None
                groundTruth = None
                for x, batch in enumerate(test):
                    data, labels = batch['X'], batch['y'].to(model.dev)
                    pred = model.predict_step(data)
                    if x == 0:
                        predictions = pred
                        groundTruth = labels
                    else:
                        predictions = torch.cat((predictions, pred), 0)
                        groundTruth = torch.cat((groundTruth, labels), 0)
                target_names = ['no irony', 'irony']
                metrics = classification_report(groundTruth.cpu().detach().numpy(), predictions.cpu().detach().numpy(),
                                                target_names=target_names, digits=4)
                print(metrics)
                myout.write(metrics + '\n')
                overallACC += accuracy_score(groundTruth.cpu().detach().numpy(), predictions.cpu().detach().numpy())
                overallF1+=fbeta_score(groundTruth.cpu().detach().numpy(), predictions.cpu().detach().numpy(), average='macro', beta=0.5)

            # ---------------------------------------------------------------------------------------------------------------------------------------------
            # Releasing the MEMORY
            torch.cuda.empty_cache()
            torch.cuda.memory_summary(device=None, abbreviated=False)
            del tokenizer
            del test
            del model
            del trainloader
            del devloader
            gc.collect()

        print(f'Overall ACC {overallACC / parameters.folds}')
        print(f'Overall F1-Macro {overallF1 / parameters.folds}')
