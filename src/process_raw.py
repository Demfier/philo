from utils.preprocess import process_raw
from models.config import model_config

if __name__ == '__main__':
    process_raw(model_config)
    print('Processed raw {} datafiles.'.format(model_config['task']))
