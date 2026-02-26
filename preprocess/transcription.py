import re
import pandas as pd
import os

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
    interventionTimeStamps = []

    repetitions = 0  
    retracings = 0   
    errors = 0  
    fragments = 0  
    unintelligibles = 0 

    for line in lines:
        if line.startswith('*PAR:'):
            processedTranscription += line[6:]
            patientLine = True
            timestamp = re.search(r'(\d+_\d+)', line)

        elif line.startswith('\t') and patientLine:
            processedTranscription += line
            timestamp = re.search(r'(\d+_\d+)', line)

        else:
            patientLine = False
        
        if line.startswith('*INV:'):
            timestamp = re.search(r'(\d+_\d+)', line)
            if timestamp:
                interventionTimeStamps.append(timestamp.group(1))

    repetitions += len(re.findall(r'\[/\]', processedTranscription))
    retracings += len(re.findall(r'\[//\]', processedTranscription))
    errors += len(re.findall(r'\[\*', processedTranscription)) 
    unintelligibles += len(re.findall(r'xxx', processedTranscription, re.IGNORECASE))

    processedTranscription = re.sub(r'&=[a-zA-Z]+', '', processedTranscription)

    fragments += len(re.findall(r'&[a-zA-Z]+', processedTranscription))

    processedTranscription = re.sub(r'\s*\d+_\d+', '', processedTranscription)
    processedTranscription = re.sub(r'\[.*?\]', '', processedTranscription)
    processedTranscription = re.sub(r'&=?[a-zA-Z]+', '', processedTranscription)
    processedTranscription = re.sub(r'\b0[a-zA-Z]+', '', processedTranscription)
    processedTranscription = re.sub(r'\(\.+\)', '', processedTranscription)
    processedTranscription = re.sub(r'\t|_', ' ', processedTranscription)
    processedTranscription = re.sub(r'\bxxx\b', '', processedTranscription, flags=re.IGNORECASE)
    processedTranscription = re.sub(r'[\d\n()&@]|\+<|<|>', '', processedTranscription)
    processedTranscription = re.sub(r'\s+\.', '', processedTranscription)
    processedTranscription = re.sub(r'\s+', ' ', processedTranscription)
    processedTranscription = re.sub(r'\s+\?', '?', processedTranscription)

    total_words=len(processedTranscription.split(' '))
    errors={
        'ID': filePath.split('/')[-1].split('.')[0],
        'repetitions': repetitions,
        'retracings': retracings,
        'errors': errors,
        'unintelligibles': unintelligibles,
        'fragments': fragments,
        'total words': total_words
    }
    df = pd.DataFrame([errors])

    if not os.path.exists('resultados'):
        os.makedirs('resultados')
    escribir_cabecera = not os.path.exists('resultados/transcriptions'+filePath.split('/')[-2]+'.csv')
    df.to_csv('resultados/transcriptions'+filePath.split('/')[-2]+'.csv', mode='a', header=escribir_cabecera, index=False)

    return processedTranscription, cleanTimeStamps(interventionTimeStamps)