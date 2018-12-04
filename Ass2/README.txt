
Prerequisites:

	- python 3.7+
	  https://www.python.org/downloads/release/python-371/

	- numpy package
	  https://pypi.org/project/numpy/
	  command: pip3 install numpy

	- pytorch
  	  command1: pip3 install        		   	 http://download.pytorch.org/whl/cu90/torch-0.4.1-cp37-cp37m-win_amd64.whl
	
	  command2: pip3 install torchvision

notes: 
	- if you use Pycharm IDE, run the aforementioned 		          	  commaneds in the terminal tag within the editor
	- if 'pip3' do not work please try only 'pip'

Run Assignemnt:

	in order to run tagger.py please put the following 	arguments in the command line (if you use Pycharm, enter 	them as parameters within 'edit configurations'):

	arg1: task (1/3/4)
	arg2: tagger type (ner/pos)	
	arg3: learning rate
	arg4: epochs (number of iterations)
	arg5: hidden layer dim
	arg6: predefined embeddings boolean: (1-with, 0-without)
		 note: in the Class NN there is diffrenitiation 			 between with/without the usage of predefined 				 embeddings (task 1/3)
	arg7: usage of sub word units (boolean: 1-yes, 0-no)

	
	example for command line run :  								
		[	python tagger1.py 1 ner 0.1 5 120 0 0     ]


files:

	following are the files of which will be created while 	running tagger1.py:
	
	for dev file:
		- accuracy_plotX.png
		- loss_plotX.png
	note: will be presented whithin partX.pdf in zip file
	
	for test file:
		- testX.pos
		- testX.ner
	
	note: X symbols the task number that was running (1/3/4)

	In addition, please use 'vocab' and 'wordVectors' data 	files in order to run the tasks (put it within the 	directory 	of the other aforementioned files)
	

	