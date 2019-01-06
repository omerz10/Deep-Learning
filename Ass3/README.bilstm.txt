
Omer Zucker: 200876548
Omer Wolf: 307965988



#####  instructions to run bilstmTrain.py  #####


User will insert 4 parameters:
* repr: 			a / b / c / d (model representation)

* train_filename: 	train filename

* model_filename: 	model filename (zip file includes data 					after training

* data_type: 		pos / ner (2 directories in the project)

command line example:

    python bilstmTrain.py a train model pos
   
    - a: 	     model representaion
    - train:   file in 'pos' directory
    - model:   zip file stores data after training
    - pos:     directory of which train file will be saved 

*** output of running will be a train file that will be stored in pos/ner directory. in addition, 2 files related to the model will also be created and will be remove while running bilstmPredict.py***


#####  instructions to run bilstmPredict.py  #####


User will insert 4 parameters:
* repr: 			a / b / c / d (model representation)

* model_filename: 	model filename (zip file includes data 					after training

* test_filename:	test filename (located in pos/ner 						directory)

* data_type: 		pos / ner (2 directories in the project)

command line example:

    python bilstmPredict.py b model test ner

    - b: 	     model representaion
    - test:   file in 'pos' directory
    - model:   zip file stores data after training
    - ner:     directory of which test file will be saved 

*** output of running will a file named 'x_results' so the x will be the representaion of the model. this file wiil be stored in pos/ner directory ***

