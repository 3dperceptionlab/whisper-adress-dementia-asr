import os
from pydub import AudioSegment
import re

def cleanTimeStamps(timeStamps):
    cleanedTimeStamps = []
    if len(timeStamps) == 0:
        return cleanedTimeStamps
    for ts in timeStamps:
        cleanedTimeStamps.append([ts.split('_')[0], ts.split('_')[1]])
    return cleanedTimeStamps

def preprocessTranscription(filePath):
    file = open(filePath, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    
    patientLine = False
    processedTranscription = ''
    patientTimeStamps = []
    transcriptionWithTimeStamps = []
    transcriptionText = ''

    for line in lines:
        if line.startswith('*PAR:'):
            if (transcriptionText!=''):
                transcriptionWithTimeStamps.append(transcriptionText)
                transcriptionText = ''

            transcriptionText += line[6:]
            patientLine = True
            timestamp = re.search(r'(\d+_\d+)', line)
            if timestamp:
                patientTimeStamps.append(timestamp.group(1))
        elif line.startswith('\t') and patientLine:
            transcriptionText += line
            timestamp = re.search(r'(\d+_\d+)', line)
            if timestamp:
                patientTimeStamps.append(timestamp.group(1))
        else:
            patientLine = False

    if transcriptionText != '':
        transcriptionWithTimeStamps.append(transcriptionText)

    for i in range(len(transcriptionWithTimeStamps)):
        transcriptionWithTimeStamps[i] = re.sub(r'\x15+', '', transcriptionWithTimeStamps[i])
        transcriptionWithTimeStamps[i] = re.sub(r'\+[./?]+', '', transcriptionWithTimeStamps[i])
        transcriptionWithTimeStamps[i] = re.sub(r'\s*\d+_\d+', '', transcriptionWithTimeStamps[i])
        transcriptionWithTimeStamps[i] = re.sub(r'\[.*?\]', '', transcriptionWithTimeStamps[i])
        transcriptionWithTimeStamps[i] = re.sub(r'&=?[a-zA-Z]+', '', transcriptionWithTimeStamps[i])
        transcriptionWithTimeStamps[i] = re.sub(r'\b0[a-zA-Z]+', '', transcriptionWithTimeStamps[i])
        transcriptionWithTimeStamps[i] = re.sub(r'\(\.+\)', '', transcriptionWithTimeStamps[i])
        transcriptionWithTimeStamps[i] = re.sub(r'\t|_', ' ', transcriptionWithTimeStamps[i])
        transcriptionWithTimeStamps[i] = re.sub(r'\bxxx\b', '', transcriptionWithTimeStamps[i], flags=re.IGNORECASE)
        transcriptionWithTimeStamps[i] = re.sub(r'[\d\n()&@<>]|\+<', '', transcriptionWithTimeStamps[i])
        transcriptionWithTimeStamps[i] = re.sub(r'\s+\.', '', transcriptionWithTimeStamps[i])
        transcriptionWithTimeStamps[i] = re.sub(r'\s+', ' ', transcriptionWithTimeStamps[i])
        transcriptionWithTimeStamps[i] = re.sub(r'\s+\?', '?', transcriptionWithTimeStamps[i])
    for line in transcriptionWithTimeStamps:
        if line.strip() != '':
            processedTranscription += line.strip() + ' '
    processedTranscription = processedTranscription.strip()


    return processedTranscription, cleanTimeStamps(patientTimeStamps), transcriptionWithTimeStamps

def processCutDataset(originalPath, originalTranscriptionsPath, processedPath):    
    chunk_size_ms = 30000

    if not os.path.exists(processedPath):
        os.makedirs(processedPath)
    
    if originalTranscriptionsPath is None:
        for filename in os.listdir(originalPath):
            if filename.endswith('.wav'):
                baseName = os.path.splitext(filename)[0]
                audioPath = os.path.join(originalPath, baseName + '.wav')
                
                original_audio = AudioSegment.from_mp3(audioPath)
                processedAudioPath = os.path.join(processedPath, baseName + '.wav')
                audio = original_audio.set_frame_rate(16000).set_channels(1)
                audio.export(processedAudioPath, format='wav')
    else:
        for filename in os.listdir(originalPath):
            if filename.endswith('.wav'):
                baseName = os.path.splitext(filename)[0]
                audioPath = os.path.join(originalPath, baseName + '.wav')
                transcriptionPath = os.path.join(originalTranscriptionsPath, baseName + '.cha')
                                
                original_audio = AudioSegment.from_mp3(audioPath)
                audio = AudioSegment.empty()

                preprocessedTranscription, patientTimeStamps, transcriptionLines = preprocessTranscription(transcriptionPath)
                
                if len(original_audio) <= chunk_size_ms:      
                    processedAudioPath = os.path.join(processedPath, baseName + '.wav')             
                    processedTranscriptionPath = os.path.join(processedPath, baseName + '.txt')
                    audioFileName = baseName + '.wav'
                    with open(processedTranscriptionPath, 'w', encoding='utf-8') as f:
                        f.write(preprocessedTranscription)
                    for timestamp in patientTimeStamps:
                        start_ms = int(timestamp[0])
                        end_ms = int(timestamp[1])
                        audio += original_audio[start_ms:end_ms]
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    audio.export(processedAudioPath, format='wav')

                else:
                    current_idx = 0
                    part = 0
                    while current_idx < len(patientTimeStamps):
                        audio=AudioSegment.empty()
                        transcription = ''
                        chunk_ms = 0
                        part_start_idx = current_idx

                        while current_idx < len(patientTimeStamps):
                            timestamp = patientTimeStamps[current_idx]
                            start_ms = int(timestamp[0])
                            end_ms = int(timestamp[1])
                            if end_ms - start_ms > chunk_size_ms:
                                current_idx+=1
                                break
                            if chunk_ms + (end_ms - start_ms) > chunk_size_ms:
                                break
                            chunk_ms += end_ms - start_ms
                            audio+=original_audio[start_ms:end_ms]
                            transcription+=transcriptionLines[current_idx].strip()+' '
                            current_idx+=1

                        if len(audio) > 0:
                            processedAudioPath = os.path.join(processedPath, f'{baseName}_part{part}.wav')   
                            processedTranscriptionPath = os.path.join(processedPath, f'{baseName}_part{part}.txt')
                            with open(processedTranscriptionPath, 'w', encoding='utf-8') as f:
                                f.write(transcription.strip())

                            audio = audio.set_frame_rate(16000).set_channels(1)
                            audio.export(processedAudioPath, format='wav')

                            part+=1

                        else:
                            if part_start_idx == current_idx:
                                current_idx += 1