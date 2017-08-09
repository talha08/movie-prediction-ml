import csv
import datetime
from extract_data import *
from word_encoder import *
# from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
###############################################################################
# Runs the various classification algorithms on the test data
###############################################################################


# setup progress bar
toolbar_width = 60
#sys.stdout.write("\n[%s" % (" " * (toolbar_width/3)))
#sys.stdout.write("***Predicting data***")
#sys.stdout.write("%s]" % (" " * (toolbar_width/3)))
sys.stdout.flush()
sys.stdout.write("\n")

# initialize variables
column = []
data_val = []
progress = 0
scores = []

#variables for calculating error margin
rf_error_margin = 0
dt_error_margin = 0
nb_error_margin = 0
svm_error_margin = 0
count = 0

# send the extracted data available from extract_data to the encode function
# this function vectorizes the text based data into ASCII format for use by
# the algorithms
encoded_data = encode(data)

# convert the float scores to int. Multiplying by 10 helps us keep the decimal
# level precision which would otherwise be lost in typecasting
i = 0
while i < len(label):
    scores.append(int (float(label[i]) * 10))
    i += 1;



# Decision Tree
dt_clf = tree.DecisionTreeClassifier()
dt_clf.fit(encoded_data, scores)



y_true = []
y_pred = []

with open('Datasets/Testing Dataset.csv') as f:

    reader = csv.DictReader(f) # read rows into a dictionary format
    i =0
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        column.append(row['movie_title'])
        column.append(row['actor_1_name'])
        column.append(row['color'])
        column.append(row['director_name'])
        column.append(row['num_critic_for_reviews'])
        column.append(row['duration'])
        column.append(row['director_facebook_likes'])
        column.append(row['actor_3_facebook_likes'])
        column.append(row['actor_2_name'])
        column.append(row['actor_1_facebook_likes'])
        column.append(row['gross'])
        column.append(row['genres'])
        column.append(row['num_voted_users'])
        column.append(row['cast_total_facebook_likes'])
        column.append(row['actor_3_name'])
        column.append(row['num_user_for_reviews'])
        column.append(row['language'])
        column.append(row['country'])
        column.append(row['budget'])
        column.append(row['title_year'])
        column.append(row['actor_2_facebook_likes'])
        column.append(row['movie_facebook_likes'])
        data_val.append(column)
        test_data = encode(data_val)

       
        #getting value for mesuare accuracy
        y_true.append(int (float(row['imdb_score']) * 10))
        y_pred.append(column)
        


        # calculate error margin for Decision Tree
        dt_error_margin += abs((dt_clf.predict (test_data)/10.0) - (float(row['imdb_score'])))

        count += 1
        column = []
        data_val = []

        


#Accuracy for Decision Tree
final_accuracy_dt = accuracy_score(y_true,  dt_clf.predict(encode(y_pred)))







# Print the error margin
print "Error margin for Decision Tree=" ,dt_error_margin/count*100, "%" 
print "Accuracy = ",final_accuracy_dt*100, "%"






#prediction
#final_pre = dt_clf.predict(encode(y_pred))
final_pre = dt_clf.predict(encode([['Date With You', 'John August', 'Color', 'Jon Gunn', '43', '90', '16', '16', 'Brian Herzlinger', '86', '85222', 'Documentary', '4285', '163', 'Jon Gunn', '84', 'English', 'USA', '1100', '2004', '23', '456']]))


#prediction print
str1 = " ".join(str(x) for x in final_pre)
print "Predicted Rating For Given Data = ", float(str1)/10








