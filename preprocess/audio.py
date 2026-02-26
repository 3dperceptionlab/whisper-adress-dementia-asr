import os
from pydub import AudioSegment
from preprocess.transcription import preprocessTranscription

def processDataset(originalPath, originalTranscriptionsPath, processedPath):    
    if not os.path.exists(processedPath):
        os.makedirs(processedPath)
    
    for filename in os.listdir(originalPath):
        if filename.endswith('.wav'):
            baseName = os.path.splitext(filename)[0]
            audioPath = os.path.join(originalPath, baseName + '.wav')
            transcriptionPath = os.path.join(originalTranscriptionsPath, baseName + '.cha')
            
            processedTranscription, interventionTimeStamps = preprocessTranscription(transcriptionPath)
            
            original_audio = AudioSegment.from_mp3(audioPath)
            audio = AudioSegment.empty()
            actual_ms = 0

            if len(interventionTimeStamps) == 0:
                audio = original_audio
            else:
                for timestamp in interventionTimeStamps:
                    start_ms = float(timestamp[0])
                    end_ms = float(timestamp[1])

                    if start_ms > actual_ms:
                        audio += original_audio[actual_ms:start_ms]
                    actual_ms = end_ms

                if actual_ms < len(original_audio):
                    audio += original_audio[actual_ms:]

            processedAudioPath = os.path.join(processedPath, baseName + '.wav')
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(processedAudioPath, format='wav')
            
            processedTranscriptionPath = os.path.join(processedPath, baseName + '.txt')
            with open(processedTranscriptionPath, 'w', encoding='utf-8') as f:
                f.write(processedTranscription)
            