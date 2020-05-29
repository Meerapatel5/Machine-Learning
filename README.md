# LearningWithNoisyLabels
Implementation of a state-of-art algorithm from the paper “Learning with Noisy Labels”[1] , which is the first one providing “guarantees for risk 
minimization under random label noise without any assumption on the true distribution.” 


Referances:
[1] “Nagarajan N, Inderjit S D, Pradeep K R, and Ambuj T. Learning with noisy labels. In Advances in neural information processing systems, 
pages 1196{1204, 2013}.”


*****Trainingmodel.py*****

Step 1:   Read requirements file
______________________________________________________________________________________________________________________
Step 2:   Call train and test files

DataSet_A needs master_test.csv & master_train.csv
DataSet_B needs master_test_b.csv & master_train_b.csv
Master files has all the samples and labels.

If you are getting an error that means you have to change the path of csv file in Trainingmodel.py at line 20 & 21.
______________________________________________________________________________________________________________________
Step 3:   Change number of features for Dataset_A and DataSet_B.

For DataSet_A
feature_size = int(42)
final_structure_feature_size = int(43)

For DataSet_B
feature_size = int(22)
final_structure_feature_size = int(23)
_______________________________________________________________________________________________________________________
step 4: Change universal class list

DataSet_A plays with 10 labels.
universal_class_list = [x for x in range(1, 11, 1)]

DataSet_B plays with 11 labels.
universal_class_list = [x for x in range(1, 12, 1)]
________________________________________________________________________________________________________________________
step5:  Save final results 

If DataSet_A is running: save the result in final_result_ [line 271 & line 286]

If DataSet_B is running: save the result in final_result_B [line 271 & line 286]
________________________________________________________________________________________________________________________


****** Let's understand the working of Trainingmodel.py******

1. Write label name of one column from A_train_label or B_train_label. One at a time starting with 'B' to 'N40'.

For example, at line 248:
test_column_name = ['B']   
test_column_error = [0]

When you get result of B column, run the Trainingmodel.py again with,
test_column_name = ['C5']
test_column_error = [0.05]

Test_column_name and test_column_error are siblings.
They need same thing from parents or they cry.


2. Here we put k = 10 in KFold verification, you can change it too. 
 Though, it will change the 'accuracy' and 'time complexity'. 


3. SVM, we are using one-vs-all multiclass classification.
 Each training label has 10 labels for DataSet_A.
 So, classifier has trained 10 times.
 1st: label 1 = 1 and other label = -1
 2nd: label 2 = 1 and other label = -1
 .....
 10th: label 10 = 1 and other label = -1


4. It will check statistical acuracy and f1_score for each label 1 to 10.

  

Thank you.



  









