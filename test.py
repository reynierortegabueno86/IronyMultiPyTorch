import zlib
import base64
import argparse
import config as conf
import sqlite3
import  hashlib
#create_cache_name=lambda _param: base64.encodebytes(zlib.compress(_param.encode("utf8"))).decode('utf8')


class SqliteDB:
    def __init__(self, dbname='database/urls.sqlite3'):
        self.db = sqlite3.connect(dbname)
        self.db.execute('''CREATE TABLE IF NOT EXISTS urls
              (hash BLOB PRIMARY KEY, url TEXT)''')

    def shorten(self, url):
        h = sqlite3.Binary(hashlib.sha256(url.encode('ascii')).digest())
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO urls VALUES (?, ?)', (h, url))
        return base64.urlsafe_b64encode(h).decode('ascii')

    def geturl(self, shortened_url):
        h = sqlite3.Binary(base64.urlsafe_b64decode(shortened_url.encode('ascii')))
        with self.db:
            url = self.db.execute('SELECT url FROM urls WHERE hash=?', (h,)).fetchone()
        if url is None:
            raise KeyError(shortened_url)
        return url[0]

    def close(self):
        if self.db!=None:
            self.db.close()



def check_params(args=None):
    parse = argparse.ArgumentParser(description='Method for automatically irony detection in multiple languages')
    parse.add_argument('-i', '--input', help='File with the input data', required=False, default="/home/reynier/PycharmProjects/IronyMultiPyTorch//data/EnRiloffEtAl13/training.tsv")
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
def replace_param_value(params, pvalues):
    info=dict(params)
    for key in pvalues:
        if key in info: info[key]=pvalues[key](info[key])
    return list(info.items())


if __name__ == "__main__":
    import sys
    parameters = check_params(sys.argv[1:])
    #print(dir(parameters))
    sql3=SqliteDB()
    #print(parameters._get_kwargs())
    par= parameters._get_kwargs()
    par = replace_param_value(par, {"input":sql3.shorten, "output":sql3.shorten})
    par.sort()

    stri="%%%".join([f'{x[0]}_{str(x[1])}' for x in par])
    print(stri)
    a=sql3.shorten(stri)
    print(a)
    print(sql3.geturl(a))
    #print(create_cache_name(stri))
    #print(parameters._get_kwargs())
    #print(parameters.input)
    sql3.close()






history.append({'loss': [], 'acc': [], 'dev_loss': [], 'dev_acc': []})
torch.cuda.empty_cache()
print('----------------------------------------------------')
#print(f'******Evaluating the FOLD {fold + 1}/{parameters.folds}')
print('----------------------------------------------------')
pretrainedmodel, tokenizer = getModelTokenizerConfig(parameters.model, parameters.language)
# Defining the data data pipeline
data_transform = TextCompose([Tokenization(tokenizer, paddingstrategy='max_length', maxlength=parameters.maxlength, truncate=True)])
# Obtaining the data split for training and test.
print("#" * 50)
print("TRAINING:", sum(trainlab), "/", len(trainlab))
print("TEST ", sum(testlab), "/", len(testlab))
print("#" * 50)
data = IronyDataModule(train=ironyDataTrain, test=ironyDataTest, batch_size=parameters.batchsize)# transform=data_transform)
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
patience = 0
batches = len(trainloader)
for epoch in range(parameters.epochs):
    if patience >= parameters.patience: break
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
        # print(torch.max(out, 1).indices.sum(), log.sum())
        history[-1]['acc'].append(acc)
        history[-1]['dev_loss'].append(dev_loss)
        history[-1]['dev_acc'].append(dev_acc)
    if model.best_measure is None or model.best_measure < dev_acc:  # here you must set your saveweights criteroin
        model.best_measure = dev_acc
        model.best_model_name = 'models/bestmodel.pt'
        model.save(model.best_model_name)
        patience = 0
    patience += 1
    print(" acc: {} ||| dev_loss: {} dev_acc: {}".format(np.round(acc, decimals=3),
                                                         np.round(dev_loss, decimals=3),
                                                         np.round(dev_acc, decimals=3)))

