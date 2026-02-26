import os
import evaluate
from datasets import Dataset, Audio, Features, Value
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers.pipelines.pt_utils import KeyDataset

def createDataset(processedPath):
    names = sorted([f[:-4] for f in os.listdir(processedPath) if f.endswith('.wav')])

    data = {
        'audio': [os.path.join(processedPath, name + '.wav') for name in names],
        'transcription': [open(os.path.join(processedPath, name + '.txt'), 'r', encoding='utf-8').read().strip() for name in names]
    }

    features = Features({
        'audio': Audio(sampling_rate=16000),
        'transcription': Value('string')
    })

    dataset = Dataset.from_dict(data, features=features)
    return dataset

def evaluateModel(model_name, dataset, no_repeat=True):
    print(f'\nEvaluando modelo: {model_name}...')  
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model=AutoModelForSpeechSeq2Seq.from_pretrained(model_name, dtype=torch_dtype).to(device)
    if model_name.split('/')[0]=='openai':
        processor = AutoProcessor.from_pretrained(model_name, language='en', task='transcribe')
    else:
        processor = AutoProcessor.from_pretrained('openai/'+model_name.split('_')[1], language='en', task='transcribe')

    pipe = pipeline(
        'automatic-speech-recognition',
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch_dtype,
        device=device,
    )

    wer = evaluate.load('wer')
    normalizer = BasicTextNormalizer()

    predictions = []
    references = [normalizer(ref) for ref in dataset['transcription']]
    
    gen_params = {
            'task': 'transcribe', 
            'language': 'en',
            'do_sample': False,
            'temperature': 0.0,
    }
    if no_repeat == True:
        gen_params['no_repeat_ngram_size'] =3
        gen_params['condition_on_prev_tokens']=True
        gen_params['compression_ratio_threshold']=1.35

    output = pipe(
        KeyDataset(dataset, 'audio'), 
        batch_size=32, 
        generate_kwargs=gen_params
    )

    for result in output:
        predictions.append(normalizer(result['text']))

    wer_score = 100 * wer.compute(references=references, predictions=predictions)
    
    return wer_score