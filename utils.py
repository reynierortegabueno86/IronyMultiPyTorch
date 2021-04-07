from transformers import AutoTokenizer, AutoModel, AutoConfig, BertTokenizer, BertModel, BertConfig, BertweetTokenizer
import zlib
import base64
import argparse
import config as conf
import sqlite3
import hashlib

modelByLanguage = {}
multilingualModel = {"mbert": "bert-base-multilingual-cased", "xlm-roberta": "xlm-roberta-large"}
spanishModel = {"beto": "dccuchile/bert-base-spanish-wwm-uncased", "spanbert": "SpanBERT/spanbert-base-uncased"}
englishModel = {"roberta": "roberta-base", "bert": "bert-base-uncased", 'bertweet': 'vinai/bertweet-base'}
italianModel = {"alberto": 'm-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0',
                'umberto': 'Musixmatch/umberto-wikipedia-uncased-v1'}

modelByLanguage["es"] = spanishModel
modelByLanguage["en"] = englishModel
modelByLanguage["it"] = italianModel
modelByLanguage["ml"] = multilingualModel

def getModelTokenizerConfig(modelName, lang):
    config, model, tokenizer = None, None, None
    if lang in modelByLanguage:
        if modelName in ["beto", 'mbert']:
            model = AutoModel.from_pretrained(
                modelByLanguage[lang][modelName])  # , output_hidden_states=True, # +return_dict=True)
            tokenizer = BertTokenizer.from_pretrained(modelByLanguage[lang][modelName])
        elif modelName in modelByLanguage[lang]:
            model = AutoModel.from_pretrained(
                modelByLanguage[lang][modelName])  # , output_hidden_states=True,return_dict=True)
            if modelName == 'bertweet':
                tokenizer = BertweetTokenizer.from_pretrained(modelByLanguage[lang][modelName])
            else:
                tokenizer = AutoTokenizer.from_pretrained(modelByLanguage[lang][modelName])
                # config = AutoConfig.from_pretrained(modelByLanguage[lang][modelName])
    return model, tokenizer

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
        if self.db != None:
            self.db.close()

def replace_param_value(params, pvalues):
    info=dict(params)
    for key in pvalues:
        if key in info: info[key]=pvalues[key](info[key])
    return list(info.items())

class BertConfig:
    def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            gradient_checkpointing=False,
            position_embedding_type="absolute",
            use_cache=True,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
