from train_evaluate.train import createDataset, train_model

if __name__ == '__main__':
    controlDataset = createDataset('30s_control')
    dementiaDataset = createDataset('30s_dementia')
    train_model('openai/whisper-tiny', controlDataset, dementiaDataset)
    train_model('openai/whisper-base', controlDataset, dementiaDataset)
    train_model('openai/whisper-small', controlDataset, dementiaDataset)
    train_model('openai/whisper-medium', controlDataset, dementiaDataset)
    train_model('openai/whisper-large', controlDataset, dementiaDataset)