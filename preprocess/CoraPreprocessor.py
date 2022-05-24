import os
import requests
import tarfile
import shutil
import argparse

DATA_URL = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
TGZ_FN = 'cora.tgz'
CITE_DATA_FN = 'cora.cites'
CONTENT_DATA_FN = 'cora.content'

DATA_DIR_DEFAULT = '../data/cora/'


class CoraPreProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not self.check_data_integrity():
            self.prepare_data()

    def check_data_integrity(self):
        """ Checks data integrity, i.e., whether we have all the raw data needed. """
        return os.path.isdir(self.data_dir) and \
               os.path.isfile(os.path.join(self.data_dir, CITE_DATA_FN)) and \
               os.path.isfile(os.path.join(self.data_dir, CONTENT_DATA_FN))

    def prepare_data(self):
        """ Prepares the data if necessary: download, extract and reposition. """
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        tgz_path = os.path.join(self.data_dir, TGZ_FN)

        if os.path.isfile(tgz_path):
            print('> Data already downloaded as %s' % tgz_path)
        else:
            print('> Downloading data from %s to %s' % (DATA_URL, tgz_path))
            response = requests.get(DATA_URL, stream=True)
            if response.status_code == 200:
                with open(tgz_path, 'wb') as f:
                    f.write(response.raw.read())

        print('> Extracting files...')
        with tarfile.open(tgz_path) as f:
            f.extractall(self.data_dir)
        temp_dir = os.path.join(self.data_dir, 'cora/')
        for file in os.listdir(temp_dir):
            shutil.move(os.path.join(temp_dir, file), self.data_dir)
        os.rmdir(temp_dir)

        print('> Data is now ready on %s' % self.data_dir)


if __name__ == '__main__':
    """ 
        Usage Example:
        python CoraPreprocessor.py -dr ../data/cora/
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-dr', '--data_dir', type=str, default=DATA_DIR_DEFAULT, help='Root directory of the input data, default = {}'.format(DATA_DIR_DEFAULT))

    FLAGS, unparsed = parser.parse_known_args()

    cora_preproc = CoraPreProcessor(data_dir=FLAGS.data_dir)
