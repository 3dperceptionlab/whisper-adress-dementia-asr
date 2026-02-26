from preprocess.audio import processDataset
from preprocess.audio_30s import processCutDataset
from preprocess.vad import vad_cut
from plots.errors_plot import error_plot
from plots.pauses_plot import pauses_plot

if __name__ == '__main__' :
    originalControl = '/dataset/train/Full_wave_enhanced_audio/cc'
    originalControlTranscriptions = '/dataset/train/transcription/cc'

    originalDementia = '/dataset/train/Full_wave_enhanced_audio/cd'
    originalDementiaTranscriptions = '/dataset/train/transcription/cd'

    originalTest = '/dataset/test/Full_wave_enhanced_audio'
    originalTestTranscriptions = '/dataset/test/transcription'

    processedControl = 'preprocessed_control'
    processedDementia = 'preprocessed_dementia'
    processedCutControl = '30s_control'
    processedCutDementia = '30s_dementia'
    processedCutTest = '30s_test'

    cutControl = '/workspace/talkbank/vad_cut_control'
    cutDementia = '/workspace/talkbank/vad_cut_dementia'

    cutControl30s = 'cut_30s_control'
    cutDementia30s = 'cut_30s_dementia'

    processDataset(originalControl, originalControlTranscriptions, processedControl)
    processDataset(originalDementia, originalDementiaTranscriptions, processedDementia)
    processCutDataset(originalControl, originalControlTranscriptions, processedCutControl)
    processCutDataset(originalDementia, originalDementiaTranscriptions, processedCutDementia)
    processCutDataset(originalTest, originalTestTranscriptions, processedCutTest)

    vad_cut(processedCutControl, cutControl30s)
    vad_cut(processedCutDementia, cutDementia30s)

    vad_cut(originalControl, cutControl)
    vad_cut(originalDementia, cutDementia)
    
    error_plot()
    pauses_plot()