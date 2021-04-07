#9, 12, 7 #NO 14
#6 NO 13, 21




# FOR ENGLISH LANGUAGE  COLECTIONS
models = ['alberto']
# FOR ENGLISH LANGUAGE  COLECTIONS
#models = ["beto", 'spanbert']
#collections = ["EnRiloffEtAl13", "EnBarbieriEtAl14", "EnPtacekEtAl14", "EnMohammadEtAl15", "EnSemEval18"]
#collections = ["EnRiloffEtAl13"]
#collections = ["EnMohammadEtAl15"]
#collections = ["EnSemEval18"]
#collections = ["EnBarbieriEtAl14", "EnPtacekEtAl14"]
#collections = ["EsIroSvA19ES","EsIroSvA19MX", "EsIroSvA19CU","EsIroSvA19"]
#fix_collection="EnMohammadEtAl15"
fix_collection="EsIroSvA19MX"
collections = ["EsIroSvA19CU", "EsIroSvA19ES"]
#collections = ["EnSemEval18", "EnRiloffEtAl13"]
#models = ["alberto"]
language=["es"]
#language=["it"]
optimizers=["adam"]
output=["output"]
#folds=[10]
batchsize=[32]
maxlength=[50]
epochs=[50]
patiences=[15]

absolutePath = "/home/rortega/PycharmProjects/IronyMultiPyTorch"
#learningStrategy=["dynamic"]
learningStrategy=["simple"]
learningRate=[5e-5]
minlearningRate=[1e-5]
lrdecay =[1e-05]
if __name__=="__main__":
    import sys
    absolutePath = sys.argv[1]
    print("#!/bin/bash")
    command = f"python3 {absolutePath}/mainTrainCvCrossDomLing.py"
    i=1
    for lang in language:
        for col in collections:
            for mod in models:
                for opt in optimizers:
                    for out in output:
                        #for f in folds:
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
                                                                print(command+f" --batchsize {batch}  --epochs {ep} " \
                                                                      f" --input {absolutePath}/data/{fix_collection}/training.tsv --language {lang} --learning {lr} --lrdecay {dec}" \
                                                                      f" --lrstategy {strat} --maxlength {leng} --minlearning {minlr} --model {mod}" \
                                                                      f" --optimizer {opt} --output {absolutePath}/{out}/{fix_collection}_{col}_{mod} --patience {pat}"\
                                                                      f" --test {absolutePath}/data/{col}/training.tsv")
                                                        else:
                                                            for minlr in minlearningRate[:1]:
                                                                print(f"echo 'running the {i} script'")
                                                                i+=1
                                                                print(command +f" --batchsize {batch}  --epochs {ep} " \
                                                                      f" --input {absolutePath}/data/{fix_collection}/training.tsv --language {lang} --learning {lr} --lrdecay {dec}" \
                                                                      f" --lrstategy {strat} --maxlength {leng} --minlearning {minlr} --model {mod}" \
                                                                      f" --optimizer {opt} --output {absolutePath}/{out}/{fix_collection}_{col}_{mod} --patience {pat}" \
                                                                      f" --test {absolutePath}/data/{col}/training.tsv")
