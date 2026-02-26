import os
import shutil
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
import re
from pydub import AudioSegment
import pandas as pd

# Applies the VAD algorithm to each file
def run_vad(file_path):

    model = Model.from_pretrained(
    'pyannote/segmentation-3.0', 
    use_auth_token=True)

    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
        'min_duration_on': 0.1,
        'min_duration_off': 0.1
    }

    pipeline.instantiate(HYPER_PARAMETERS)

    return pipeline(file_path)

# Apply the VAD algorithm to every file in a folder and create new ones without silences
def vad_cut(file_path, cut_path):
    pauses = []
    if not os.path.exists(cut_path):
        os.makedirs(cut_path)

    for filename in os.listdir(file_path):
        if filename.endswith('.wav'):
            baseName = os.path.splitext(filename)[0]
            audioPath = os.path.join(file_path, baseName + '.wav')
            outputPath = os.path.join(cut_path, baseName + '.wav')

            transcriptionPath = os.path.join(file_path, baseName + '.txt')
            outputTranscriptionPath = os.path.join(cut_path, baseName + '.txt')
            if os.path.isfile(transcriptionPath):
                shutil.copy(transcriptionPath, outputTranscriptionPath)
           
            original_audio = AudioSegment.from_wav(audioPath) 
            vad_audio = AudioSegment.empty()

            vad_lines = list(run_vad(audioPath).get_timeline().support())

            actual_ms=0
            for line in vad_lines:
                start_ms = int(line.start * 1000)
                end_ms = int(line.end * 1000)
                vad_audio += original_audio[start_ms:end_ms]

                pause={
                    'ID': baseName,
                    'Pause duration': (start_ms-actual_ms)/1000
                }

                pauses.append(pause)
                actual_ms=end_ms
            
            if actual_ms < len(original_audio):
                pause={
                    'ID': baseName,
                    'Pause duration': (len(original_audio)-actual_ms)/1000
                }

                pauses.append(pause)
                
            vad_audio.export(outputPath, format='wav')

    if not os.path.exists('resultados'):
        os.makedirs('resultados')
    df = pd.DataFrame(pauses)
    df.to_csv('resultados/'+file_path.split('/')[-1]+'.csv', index=False)