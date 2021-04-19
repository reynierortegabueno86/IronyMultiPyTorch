#9, 12, 7 #NO 14
#6 NO 13, 21

#MULTILINGUAL MODELS
#ITALIAN  GPU6
#SPANISH  GPU9
#INGLES  GPU13







# FOR ENGLISH LANGUAGE  COLECTIONS
models = ["roberta", 'bertweet', 'bert']
models = ["beto", 'spanbert']
models = ["alberto", 'umberto']
models=["mbert", "xlm-roberta"]
models=["mbert"]# "xlm-roberta"]

# FOR ENGLISH LANGUAGE  COLECTIONS
#collections = ["EnRiloffEtAl13", "EnBarbieriEtAl14", "EnPtacekEtAl14", "EnMohammadEtAl15", "EnSemEval18"]
#collections = ["EnRiloffEtAl13"]
#collections = ["EnMohammadEtAl15"]
#collections = ["EnSemEval18"]
collections = ["EnBarbieriEtAl14", "EnPtacekEtAl14"]
#collections = ["EsIroSvA19ES","EsIroSvA19MX", "EsIroSvA19CU","EsIroSvA19"]
#collections = ["ItIronITA18","ItIronITAtwitiro18", "ItIronITAother18"]
#collections=["EnRiloffEtAl13", "EnMohammadEtAl15", "EnSemEval18"]
#language=["en"]
language=["ml"]
optimizers=["adam", "rmsprop"]
output=["output"]
folds=[10]
batchsize=[32]
maxlength=[50]
epochs=[50]
patiences=[15]

absolutePath = "/home/rortega/PycharmProjects/IronyMultiPyTorch"
#learningStrategy=["simple", "dynamic"]
learningStrategy=[ "dynamic"]
learningRate=[5e-5]
minlearningRate=[1e-5]
lrdecay =[1e-05]
if __name__=="__main__":
    import sys
    absolutePath = sys.argv[1]
    print("#!/bin/bash")
    command = f"python3 {absolutePath}/mainTrainCvMonoLing.py"
    i=1
    for lang in language:
        for col in collections:
            for mod in models:
                for opt in optimizers:
                    for out in output:
                        for f in folds:
                            for leng in maxlength:
                                for batch in batchsize:
                                    for ep in epochs:
                                        for pat in patiences:
                                            for dec in lrdecay:
                                                for strat in learningStrategy:
                                                    for lr in learningRate:
                                                        if strat=="dynamic":
                                                            for minlr in minlearningRate:
                                                                print(f"echo 'running the {i} script'")
                                                                i+=1
                                                                print(command+f" --batchsize {batch}  --epochs {ep} --folds {f} " \
                                                                      f"--input {absolutePath}/data/{col}/training.tsv --language {lang} --learning {lr} --lrdecay {dec}" \
                                                                      f" --lrstategy {strat} --maxlength {leng} --minlearning {minlr} --model {mod}" \
                                                                      f" --optimizer {opt} --output {absolutePath}/{out}/{lang}_{col} --patience {pat}")
                                                        else:
                                                            for minlr in minlearningRate[:1]:
                                                                print(f"echo 'running the {i} script'")
                                                                i+=1
                                                                print(command +f" --batchsize {batch}  --epochs {ep} --folds {f} " \
                                                                      f"--input {absolutePath}/data/{col}/training.tsv --language {lang} --learning {lr} --lrdecay {dec}" \
                                                                      f" --lrstategy {strat} --maxlength {leng} --minlearning {minlr} --model {mod}" \
                                                                      f" --optimizer {opt} --output {absolutePath}/{out}/{col} --patience {pat}")




