from train_evaluate.evaluate import (
    createDataset,
    evaluateModel
)
import json
import numpy as np
import glob

n_folds=5
ds = createDataset('30s_test')
tiny = 'openai/whisper-tiny'
base = 'openai/whisper-base'
small  = 'openai/whisper-small'
medium  = 'openai/whisper-medium'
large  = 'openai/whisper-large'
models = [tiny, base, small, medium, large]

def evaluateBaseline():
    results = {}
    for model in models:
        results[model] = evaluateModel(model, ds)
    with open(f'resultados/Test_Baseline.json', 'w') as f:
        json.dump(results, f, indent=4)

def evaluateCV():
    for model in models:
        model_wer=[]
        for i in range(n_folds):
            fold=str(i+1)
            fold_name='whisper_'+ model.split('/')[1]+'_fold'+fold
            fold_dir = glob.glob(fold_name+'/*')[0]
            model_wer.append(evaluateModel(fold_dir, ds))
        model_results = {
            'avg_wer': sum(model_wer) / n_folds,
            'std_wer': np.std(model_wer),
            'history_wer': model_wer,
        }
        with open(f'resultados/Test_'+model.split('/')[1]+'.json', 'w') as f:
            json.dump(model_results, f, indent=4)   