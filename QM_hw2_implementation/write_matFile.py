import scipy.io as sio

adict_train = {}
adict_test = {}
adict_eval = {}

adict_train['feature'] = []
adict_train['label'] = []

adict_test['feature'] = []
adict_test['label'] = []

adict_eval['feature'] = []

selected_features = [2, 6, 11, 15, 21, 26, 34, 37, 41, 42, 43]
count = 0
with open('SmarterML_Training.Input','r') as f:
    for line in f:
        feature = []
        fea_index = 1
        for word in line.split():
            value = [float(word)]
            if fea_index in selected_features:
                feature.append(value)
            fea_index = fea_index + 1
        #if count < 1000:
        adict_train['feature'].append(feature)
        #else:
        #    adict_test['feature'].append(feature)
        count = count+1        

count = 0
with open('SmarterML_Training.Label','r') as f:
    for line in f:
        label = []
        for word in line.split():
            value = [float(word)]
            label.append(value)
        #if count < 1000:
        adict_train['label'].append(label)
        #else:
        #    adict_test['label'].append(label)
        count = count+1

with open('SmarterML_Eval.Input','r') as f:
    for line in f:
        feature = []
        fea_index = 1
        for word in line.split():
            value = [float(word)]
            #if fea_index in selected_features:
            feature.append(value)
            fea_index = fea_index + 1            
        adict_eval['feature'].append(feature)

sio.savemat('SmarterML_Training_1250_SF.mat', adict_train)
sio.savemat('SmarterML_testing_250_SF.mat', adict_test)
sio.savemat('SmarterML_eval.mat', adict_eval)