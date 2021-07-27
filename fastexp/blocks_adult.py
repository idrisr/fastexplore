from fastai.tabular.all import *
path = untar_data(URLs.ADULT_SAMPLE)

dls = TabularDataLoaders.from_csv(path/"adult.csv", path=path, y_names='salary',
        cat_names=["workclass","education","marital-status","occupation","relationship","race"],
        cont_names=["age","fnlwgt","education-num"],
        procs=[Categorify, FillMissing, Normalize])
