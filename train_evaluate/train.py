from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset, Audio, Features, Value
from sklearn.model_selection import KFold
from transformers import (
    WhisperTokenizer,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from collections import Counter
import os, json, torch, evaluate, random, gc
import numpy as np

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
processor = None
tokenizer = None
normalizer = BasicTextNormalizer()
metric = evaluate.load('wer')

def createDataset(processedPath):
    names=sorted([f[:-4] for f in os.listdir(processedPath) if f.endswith('.wav')])

    data = {
        'audio': [os.path.join(processedPath, name + '.wav') for name in names],
        'transcription': [open(os.path.join(processedPath, name + '.txt'), 'r', encoding='utf-8').read().strip() for name in names],
        'patient_id': [name.split('_')[0] for name in names]
    }

    features = Features({
        'audio': Audio(sampling_rate=16000),
        'transcription': Value('string'),
        'patient_id': Value('string')
    })

    dataset = Dataset.from_dict(data, features=features)
    dataset = dataset.filter(lambda x : len(x['audio']['array']) >= 16000*3)
    return dataset

def prepare_dataset(batch):
    audio_arrays = [x['array'] for x in batch['audio']]
    inputs = processor.feature_extractor(audio_arrays, sampling_rate=16000, return_tensors='pt', padding='max_length')
    label_features = processor(text=batch['transcription'], return_tensors='pt', padding=True, return_attention_mask=True)
    labels = label_features.input_ids
    labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
    inputs['labels'] = labels
    return inputs

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
 
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_features': feature['input_features']} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')
 
        label_features = [{'input_ids': feature['labels']} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt')
 
        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)
 
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
 
        batch['labels'] = labels
 
        return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
 
    label_ids[label_ids == -100] = tokenizer.pad_token_id
 
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    references=[normalizer(ref) for ref in label_str]
    predictions=[normalizer(pred) for pred in pred_str]
    wer = 100 * metric.compute(predictions=predictions, references=references)
 
    return {'wer': wer}

def train_model(model_id, dataset1, dataset2):
    n_folds=5
    epochs = 10

    if 'medium' in model_id or 'large' in model_id:
        batch_size = 4
    else:
        batch_size = 16

    global tokenizer
    tokenizer = WhisperTokenizer.from_pretrained(model_id, language='en', task='transcribe')
    global processor
    processor = WhisperProcessor.from_pretrained(model_id, language='en', task='transcribe')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    unique_control = np.array(sorted(list(set(dataset1['patient_id']))))
    unique_dementia = np.array(sorted(list(set(dataset2['patient_id']))))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=123)

    fold_control=[]
    fold_dementia=[]
    fold_combined=[]

    splits_control = list(kf.split(unique_control))
    splits_dementia = list(kf.split(unique_dementia))

    for fold, ((train_idx_c, test_idx_c), (train_idx_d, test_idx_d)) in enumerate(zip(splits_control, splits_dementia)):
        train_ids = np.concatenate([unique_control[train_idx_c], unique_dementia[train_idx_d]])
        test_ids = np.concatenate([unique_control[test_idx_c], unique_dementia[test_idx_d]])

        ds_train = concatenate_datasets([
            dataset1.filter(lambda x: x['patient_id'] in unique_control[train_idx_c]),
            dataset2.filter(lambda x: x['patient_id'] in unique_dementia[train_idx_d])
        ])
        ds_test_control = dataset1.filter(lambda x: x['patient_id'] in unique_control[test_idx_c])
        ds_test_dementia = dataset2.filter(lambda x: x['patient_id'] in unique_dementia[test_idx_d])
        ds_test_combined = concatenate_datasets([ds_test_control, ds_test_dementia])

        ds_train = ds_train.map(prepare_dataset, remove_columns=ds_train.column_names, batched=True).shuffle(seed=42)
        ds_test_control = ds_test_control.map(prepare_dataset, remove_columns=ds_test_control.column_names, batched=True)
        ds_test_dementia = ds_test_dementia.map(prepare_dataset, remove_columns=ds_test_dementia.column_names, batched=True)
        ds_test_combined = ds_test_combined.map(prepare_dataset, remove_columns=ds_test_combined.column_names, batched=True)

        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        model.to(device)
        
        model.generation_config.forced_decoder_ids = None
        model.generation_config.language = 'en'
        model.generation_config.task = 'transcribe'
        model.generation_config.no_repeat_ngram_size = 3
        model.generation_config.condition_on_prev_tokens = True
        model.generation_config.compression_ratio_threshold = 1.35
        model.config.dropout = 0.1
        model.config.attention_dropout = 0.1
        model.config.apply_spec_augment = True
        model.config.mask_time_prob= 0.05
        model.config.mask_feature_prob = 0.05

        out_dir = f"whisper_{model_id.split('/')[-1]}_fold{fold+1}"

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=64 // batch_size,
            learning_rate=1.25e-5,
            lr_scheduler_type='linear',
            warmup_steps=20,
            weight_decay=0.02,
            bf16=True,
            fp16=False,
            num_train_epochs=epochs,
            eval_strategy='epoch',
            logging_strategy='epoch',
            save_strategy='epoch',
            predict_with_generate=True,
            generation_max_length=225,
            report_to=['none'],
            load_best_model_at_end=True,
            metric_for_best_model='wer',
            greater_is_better=False,
            dataloader_num_workers=8,
            save_total_limit=1,
            seed=42,
            data_seed=42,

            **( 
                {
                    'optim': 'adamw_bnb_8bit',
                    'gradient_checkpointing': True,
                    'gradient_checkpointing_kwargs': {'use_reentrant': False}
                } if 'large' in model_id else {} 
            )
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=ds_train,
            eval_dataset=ds_test_combined,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=processor.feature_extractor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )

        torch.cuda.empty_cache()
        trainer.train()


        metrics_c = trainer.evaluate(eval_dataset=ds_test_control, metric_key_prefix='control')
        metrics_d = trainer.evaluate(eval_dataset=ds_test_dementia, metric_key_prefix='dementia')
        metrics_all = trainer.evaluate(eval_dataset=ds_test_combined, metric_key_prefix='combined')

        fold_control.append(metrics_c['control_wer'])
        fold_dementia.append(metrics_d['dementia_wer'])
        fold_combined.append(metrics_all['combined_wer'])
        

        with open(f"resultados/fold{fold+1}_{model_id.replace('/','_')}.json", 'w') as f:
            json.dump({**metrics_c, **metrics_d, **metrics_all}, f, indent=4)

        del model
        del trainer
        del ds_train
        del ds_test_combined
        torch.cuda.empty_cache()
        gc.collect()

    avg_results = {
        'avg_control_wer': sum(fold_control) / n_folds,
        'avg_dementia_wer': sum(fold_dementia) / n_folds,
        'avg_combined_wer': sum(fold_combined) / n_folds,
        'std_combined_wer': np.std(fold_combined),
        'history_control': fold_control,
        'history_dementia': fold_dementia
    }

    with open(f"resultados/FINAL_CV_{model_id.replace('/','_')}.json", 'w') as f:
        json.dump(avg_results, f, indent=4)