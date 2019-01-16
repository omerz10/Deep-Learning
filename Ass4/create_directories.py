import os


project_dirctory = os.path.dirname(os.path.realpath(__file__))
data_dirctory = os.path.join(project_dirctory, 'data')

# directory doesn't exist yet - create sub directories
if not os.path.exists(data_dirctory):
    os.makedirs(data_dirctory)
    snli_dirctory = os.path.join(data_dirctory, 'snli')
    os.makedirs(snli_dirctory)
    glove_dirctory = os.path.join(data_dirctory, 'glove')
    os.makedirs(glove_dirctory)
    vocabulary_directory = os.path.join(data_dirctory, 'vocabulary')
    os.makedirs(vocabulary_directory)
print('directory is already exists')