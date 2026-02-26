from train_evaluate.evaluate import createDataset, evaluateModel
from datasets import concatenate_datasets
import json

if __name__ == '__main__':
    defaultControlDataset = createDataset('30s_control')
    defaultDementiaDataset = createDataset('30s_dementia')
    combinedDefaultDataset = concatenate_datasets([defaultControlDataset, defaultDementiaDataset])

    cutControlDataset = createDataset('cut_30s_control')
    cutDementiaDataset = createDataset('cut_30s_dementia')
    combinedCutDataset = concatenate_datasets([cutControlDataset, cutDementiaDataset])

    tiny = 'openai/whisper-tiny'
    base = 'openai/whisper-base'
    small  = 'openai/whisper-small'
    medium  = 'openai/whisper-medium'
    large  = 'openai/whisper-large'
    models = [tiny, base]

    results = {}
    for model in models:
        results[model] = {
            'default_combined': evaluateModel(model, combinedDefaultDataset, False),
            'vad_combined': evaluateModel(model, combinedCutDataset, False),
            'no_repeat_control': evaluateModel(model, defaultControlDataset),
            'no_repeat_dementia': evaluateModel(model, defaultDementiaDataset),
            'no_repeat_combined': evaluateModel(model, combinedDefaultDataset),
        } 

    with open(f'resultados/Train_Baseline.json', 'w') as f:
        json.dump(results, f, indent=4)
