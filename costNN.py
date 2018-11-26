from sklearn.svm import SVC 

def costNN(position,traininputs,trainoutputs,testinputs,testoutputs):
    dim=len(testinputs[0])
    index=[0]
    train_input = []
    test_input = []
    for i in range(1,dim):
        if position[i]>0.3:
            index.append(i)
    
    print(index)
    N=len(index)
            
    for i in range(0,len(traininputs)):
        traininp=traininputs[i]
        row=[]
        for j in range(0,len(index)):
            row.append(traininp[index[j]])
        train_input.append(row)
    
    for i in range(0,len(testinputs)):
        testinp=testinputs[i]
        row=[]
        for j in range(0,len(index)):
            row.append(testinp[index[j]])
        test_input.append(row)
    
    #print(N)
    #print(len(train_input[0]))

    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(train_input, trainoutputs)  
      
    # model accuracy for X_test   
    accuracy = svm_model_linear.score(test_input, testoutputs) 

    print(accuracy)
    #print(0.8*(accuracy) + 0.2*((dim-N)/(dim*1.0)))
    return 0.8*(accuracy) + 0.2*((dim-N)/(dim*1.0)); 