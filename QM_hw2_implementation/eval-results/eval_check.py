import scipy.io as sio

source = open('eval_Logistic_Regression.txt','r')
target = open('eval_SF_ensemble_boosted_trees.txt','r')
num_sample = 1750
count = 0
for i in range(0, num_sample):
    s = source.readline()
    t = target.readline()
    if s==t:
        count = count+1
accuracy = count/num_sample
print(accuracy)
print(count)


