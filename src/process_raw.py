from utils.preprocess_utils import process_raw
from models.config import config

if __name__ == '__main__':
    process_raw(config)
    print('Processed raw {} datafiles.'.format(config['dataset']))
