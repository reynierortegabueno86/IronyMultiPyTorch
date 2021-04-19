import pandas as pd
import numpy as np
import os
import re
from utils import SqliteDB

class ResultsAnalizer:
    def __init__(self, path, dirDB, name="irony"):
        assert os.path.isdir(path) == True
        assert os.path.isdir(dirDB)==True
        self.db = SqliteDB(dbname=dirDB + os.sep + "urls.sqlite3")
        self.experiments = [x for x in os.listdir(path) if not os.path.isdir(x)]
        self.path = path
        self.name = name

    def _getParameter4Experiment(self, filename):
        return [tuple(x.rsplit("_", 1)) for x in self.db.geturl(filename).split("%%%")]

    def _read_experiment(self, expLines):
        print(len((expLines)))
        assert len(expLines) == 6
        pos = expLines[2][1:]
        neg = expLines[1][2:]
        micro = expLines[5][2:]
        macro = expLines[4][2:]
        metrics = [[f"{self.name}_Pr", float(pos[0])], [f"{self.name}_Rc", float(pos[1])],
                   [f"{self.name}_F1", float(pos[2])],
                   [f"{self.name}_Supp", int(pos[3])],
                   [f"No{self.name}_Pr", float(neg[0])], [f"No{self.name}_Rc", float(neg[1])],
                   [f"No{self.name}_F1", float(neg[2])],
                   [f"No{self.name}_Supp", int(neg[3])],
                   ["WAvg_Pr", float(micro[0])], ["WAvg_Rc", float(micro[1])], ["WAvg_F1", float(micro[2])],
                   ["Macro_Pr", float(macro[0])], ["Macro_Rc", float(macro[1])], ["Macro_F1", float(macro[2])]]
        return metrics

    def _read_experiment_cv(self, _path):
        metrics = [[f"{self.name}_Pr", 0], [f"No{self.name}_Rc", 0], [f"No{self.name}_F1", 0],
                   [f"No{self.name}_Supp", 0],
                   [f"No{self.name}_Pr", 0],
                   [f"No{self.name}_Rc", 0], [f"No{self.name}_F1", 0], [f"No{self.name}_Supp", 0], ["WAvg_Pr", 0],
                   ["WAvg_Rc", 0],
                   ["WAvg_F1", 0], ["Macro_Pr", 0], ["Macro_Rc", 0], ["Macro_F1", 0]]

        with open(_path) as myfile:
            lines = [re.sub('\s+', ' ', line) for line in myfile.readlines()]
            lines = [x.strip().split(" ") for x in lines if x.strip() != ""]
            l = len(lines)
            cv = l // 6
            if l % 6 != 0: return None
            lines = np.asarray(lines)
            lines = lines.reshape(cv, 6)
            for block in lines:
                metr_exp = self._read_experiment(block)
                print(metr_exp)
                for i in range(len(metrics)):
                    metrics[i][1] += metr_exp[i][1]
            for i in range(len(metrics)):
                metrics[i][1] = metrics[i][1] / cv
        return metrics

    def summary_experimental_results(self, out="results.xlsx"):
        mnames, pnames, dataframe, i = [], [], [], 1
        for f in self.experiments:
            params = self._getParameter4Experiment(f)
            metrics = self._read_experiment_cv(self.path + os.sep + f)
            if metrics != None:
                pvalues, pnames = [x[1] for x in params], [x[0] for x in params]
                mvalues, mnames = [x[1] for x in metrics], [x[0] for x in metrics]
                dataframe.append(pvalues + mvalues)
        # mnames = [x.replace("Humor", name) for x in mnames]
        df = pd.DataFrame(dataframe, columns=pnames + mnames)
        df.to_excel(out, index=False)


resAnalyzerMoh = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/English/EnMohammadEtAl15/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/English/EnMohammadEtAl15/database/",
    name='irony')

#resAnalyzerMoh.summary_experimental_results("/tmp/MohammadEtAl15.xlsx")


resAnalyzerRil = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/English/EnRiloffEtAl13/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/English/EnRiloffEtAl13/database/",
    name='irony')
#resAnalyzerRil.summary_experimental_results("/tmp/RiloffEtAl13.xlsx")

resAnalyzerSem = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/English/EnSemEval18/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/English/EnSemEval18/database/",
    name='irony')
#resAnalyzerSem.summary_experimental_results("/tmp/SemEvalEtAl18.xlsx")


resAnalyzerMoh = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/English/BarbieriEtAl14/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/English/BarbieriEtAl14/database/",
    name='irony')

#resAnalyzerMoh.summary_experimental_results("/tmp/BarbieriEtAl14.xlsx")

resAnalyzerMoh = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/English/PtacekEtAl2014/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/English/PtacekEtAl2014/database/",
    name='irony')

resAnalyzerMoh.summary_experimental_results("/tmp/EnPtacekEtAl2014.xlsx")





resAnalyzerSem = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/Spanish/EsIroSvA19/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/Spanish/EsIroSvA19/database/",
    name='irony')
#resAnalyzerSem.summary_experimental_results("/tmp/EsIroSvA19.xlsx")


resAnalyzerSem = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/Spanish/EsIroSvA19CU/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/Spanish/EsIroSvA19CU/database/",
    name='irony')
#resAnalyzerSem.summary_experimental_results("/tmp/EsIroSvA19CU.xlsx")

resAnalyzerSem = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/Spanish/EsIroSvA19ES/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/Spanish/EsIroSvA19ES/database/",
    name='irony')
#resAnalyzerSem.summary_experimental_results("/tmp/EsIroSvA19ES.xlsx")


resAnalyzerSem = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/Spanish/EsIroSvA19MX/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/Spanish/EsIroSvA19MX/database/",
    name='irony')
#resAnalyzerSem.summary_experimental_results("/tmp/EsIroSvA19MX.xlsx")

resAnalyzerSem = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/Italian/ItIronITA18/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/Italian/ItIronITA18/database/",
    name='irony')
#resAnalyzerSem.summary_experimental_results("/tmp/ItIronITA2018.xlsx")


resAnalyzerSem = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/Italian/ItIronITAtwitiro18/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/Italian/ItIronITAtwitiro18/database/",
    name='irony')
#resAnalyzerSem.summary_experimental_results("/tmp/ItIronITAtwitiro18.xlsx")


resAnalyzerSem = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/Italian/ItIronITAother18/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/Monolingual/Italian/ItIronITAother18/database/",
    name='irony')
#resAnalyzerSem.summary_experimental_results("/tmp/ItIronITAother18.xlsx")


resAnalyzerSem = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/MonoMultiMultiModel/English/ALL/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/MonoMultiMultiModel/English/database/",
    name='irony')
#resAnalyzerSem.summary_experimental_results("/tmp/mlEnglishAll.xlsx")

resAnalyzerSem = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/MonoMultiMultiModel/Spanish/ALL/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/MonoMultiMultiModel/Spanish/database/",
    name='irony')
#resAnalyzerSem.summary_experimental_results("/tmp/mlSpanishAll.xlsx")

resAnalyzerSem = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/MonoMultiMultiModel/Italian/ALL/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/MonoMultiMultiModel/Italian/database/",
    name='irony')
#resAnalyzerSem.summary_experimental_results("/tmp/mlItalianAll.xlsx")














#############################################################################################3
#######CROSS-DOMAINS##########################################################3
resAnalyzerSem = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/CrossDomain/English/ALL/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/CrossDomain/English/database/",
    name='irony')
resAnalyzerSem.summary_experimental_results("/tmp/EnCrossDomains.xlsx")

import sys
sys.exit()

resAnalyzerSem = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/CrossDomain/Spanish/ALL/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/CrossDomain/Spanish/database/",
    name='irony')
#resAnalyzerSem.summary_experimental_results("/tmp/EsCrossDomains.xlsx")

resAnalyzerSem = ResultsAnalizer(
    path="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/CrossDomain/Italian/ALL/",
    dirDB="/media/reynier/DATA/Figurative Language Research/BiAS IRONY/Experiments/CrossDomain/Italian/database/",
    name='irony')

#resAnalyzerSem.summary_experimental_results("/tmp/ItCrossDomains.xlsx")