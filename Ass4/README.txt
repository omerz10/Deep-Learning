****************************Authors****************************

Omer Zucker     200876548

Omer Wolf       307965988

*************************Prerequisites**************************

- python 3.7
- pip3
- numpy package
- pytorch
- nltl

how to install:

python 3.7:
        https://www.python.org/downloads/release/python-372/
pip3:
        linux: https://askubuntu.com/questions/778052/installing-pip3-for-python3-on-ubuntu-16-04-lts-using-a-proxy
        windows: https://stackoverflow.com/questions/41501636/how-to-install-pip3-on-windows
pytorch:
        https://pytorch.org/

pip3 install numpy
pip3 install nltk
python -c "import nltk; nltk.download('punkt')"


****************************Data*****************************

1 run create_directories.py for creating data directories in your project.
  three directories will be opened:
      - snli
      - glove
      - vocabulary

2. download the following zip files from The Stanford Natural Language Inference (SNLI):

    a. 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
        after download, insert all json files from snli_1.0 into your new directory "data/snli"
    'http://www-nlp.stanford.edu/data/glove.840B.300d.zip'
         after download, insert glove.840B.300d.txt into your new directory "data/glove"

3. run extract_data.py for import all data from downloaded files and save it by a vocabulary within its directory

************************train/test model************************

1. run train_nli.py
    [*** important note: the running time of this model by laptop can take 24-72 hours !!! depending on your computer/server.
    to make it faster, run it by a machine with GPU (and add relevant arguments to config)*** ]

2. run test_nli.py


<END>