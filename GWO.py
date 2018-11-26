import random
import numpy
import costNN as cost

def GWO(dim,SearchAgents_no,Max_iter,trainInput,trainOutput,testInput,testOutput):
       
    # initialize alpha, beta, and delta_pos
    Alpha_pos=numpy.zeros(dim)
    Alpha_score=float(0)
    
    Beta_pos=numpy.zeros(dim)
    Beta_score=float(0)
    
    Delta_pos=numpy.zeros(dim)
    Delta_score=float(0)
    
    #Initialize the positions of search agents
    Positions=numpy.random.uniform(0,1,(SearchAgents_no,dim))

    # Main loop
    for l in range(0,Max_iter):
        for i in range(0,SearchAgents_no):
            
            # Return back the search agents that go beyond the boundaries of the search space
            Positions[i,:]=numpy.clip(Positions[i,:], 0, 1)
          
            # Calculate objective function for each search agent
            fitness=cost.costNN(Positions[i,:],trainInput,trainOutput,testInput,testOutput)
            
            # Update Alpha, Beta, and Delta
            if fitness>Alpha_score :
                Delta_score=Beta_score
                Delta_pos=Beta_pos;
                Beta_score=Alpha_score
                Beta_pos=Alpha_pos;
                Alpha_score=fitness; # Update alpha
                Alpha_pos=Positions[i,:]
            
            
            if (fitness<Alpha_score and fitness>Beta_score ):
                Delta_score=Beta_score
                Delta_pos=Beta_pos;
                Beta_score=fitness  # Update beta
                Beta_pos=Positions[i,:]
            
            
            if (fitness<Alpha_score and fitness<Beta_score and fitness>Delta_score): 
                Delta_score=fitness # Update delta
                Delta_pos=Positions[i,:]
        
        #print(Alpha_score)
        #print(Beta_score)
        #print(Delta_score)
            
        a=2-l*((2)/Max_iter); # a decreases linearly fron 2 to 0
        #print(a)
        
        # Update the Position of search agents including omegas
        for i in range(0,SearchAgents_no):
            for j in range (0,dim):     
                           
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
                
                A1=2*a*r1-a; # Equation (3.3)
                C1=2*r2; # Equation (3.4)
                
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]); # Equation (3.5)-part 1
                X1=Alpha_pos[j]-A1*D_alpha; # Equation (3.6)-part 1
                           
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a; # Equation (3.3)
                C2=2*r2; # Equation (3.4)
                
                D_beta=abs(C2*Beta_pos[j]-Positions[i,j]); # Equation (3.5)-part 2
                X2=Beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2       
                
                r1=random.random()
                r2=random.random() 
                
                A3=2*a*r1-a; # Equation (3.3)
                C3=2*r2; # Equation (3.4)
                
                D_delta=abs(C3*Delta_pos[j]-Positions[i,j]); # Equation (3.5)-part 3
                X3=Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3             
                
                Positions[i,j]=(X1+X2+X3)/3  # Equation (3.7)
    print("-------------------------")
    print("-------------------------")
    print("-------------------------")
    print("-------------------------")
    print("The accuracy of test set & optimal features selected are:")
    #print(Alpha_pos)
    fitness=cost.costNN(Alpha_pos,trainInput,trainOutput,testInput,testOutput)
    
    return Alpha_pos
    

