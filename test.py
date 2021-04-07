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
