import numpy as np
import time  

_m_at_eq         = 589
_m_ocean_eq      = 900+37100 
#_m_land_eq       = 2500

_m_land_eq       = 550

# (450+650)/2+(1500+2400)/2

_m_ocean_land_eq = _m_ocean_eq + _m_land_eq

_bounds={
    'm_L1':[(550/1.0,550*1.001)],
    'm_O1':[(900/1.0,900*1.001)],
    'a_level_1':[(1e-6,.15)],
    'a_level_2':[(1e-6,.15)],
    'a_level_3':[(1e-6,.15)],
}

def model_cdice(a=np.empty(0),m=np.empty(0)): 
    global _bounds
    global _m_at_eq
    global _m_ocean_eq
    p = 3
    if(len(a)==0 and len(m)==0):
        a_size = 2
        m_size = 1

        info            = {}
        info['p']       = p
        info['name']    = 'CDICE' 
        info['m_names'] = ['AT','O_1','O_2']
        
        info['a_bounds'] = _bounds['a_level_1'] + _bounds['a_level_2']
        info['m_bounds'] = _bounds['m_O1']
         
        return [a_size,m_size,info]
    
    ##################################
    # Assign variable
    ##################################
    A_21     = a[0] #AT-X_1
    A_32     = a[1] #X_1-[O+L]
    
    m_1 = 607 #AT
    m_2 = m[0] #X_1
    m_3 = m[1] #O+L
    
    ##################################
    # Generate matrix
    ##################################
    A = np.zeros(shape=(p,p))
    
    A[1,0] = A_21
    A[2,1] = A_32
    
    A[0,1] = (A_21*m_1)/m_2
    A[1,2] = (A_32*m_2)/m_3

    temp = -np.sum(A,0)
    for i in range(0,p):
        A[i,i] = temp[i]
        
    m_eq = np.array([m_1,m_2,m_3])
    x_vec = np.array(list(a)+list(m))

    return [A,m_eq,x_vec]


###########################################
# AT -> X_1 -> [X_2]
###########################################
def model_3sr(a=np.empty(0),m=np.empty(0)): 
    global _bounds
    global _m_at_eq
    global _m_ocean_eq
    p = 3
    
    if(len(a)==0 and len(m)==0):
        a_size = 2
        m_size = 1

        info            = {}
        info['p']       = p
        info['name']    = '3SR' 
        info['m_names'] = ['AT','O_1','O_2']
        
        info['a_bounds'] = _bounds['a_level_1'] + _bounds['a_level_2']
        info['m_bounds'] = _bounds['m_O1']
         
        return [a_size,m_size,info]
    
    ##################################
    # Assign variable
    ##################################
    A_21     = a[0] #AT-X_1
    A_32     = a[1] #X_1-[O+L]
    
    m_1 = _m_at_eq                       #AT
    m_2 = m[0]                           #X_1
    m_3 = _m_ocean_eq - m_2         #O+L
    
    ##################################
    # Generate matrix
    ##################################
    A = np.zeros(shape=(p,p))
    
    A[1,0] = A_21
    A[2,1] = A_32
    
    A[0,1] = (A_21*m_1)/m_2
    A[1,2] = (A_32*m_2)/m_3

    temp = -np.sum(A,0)
    for i in range(0,p):
        A[i,i] = temp[i]
        
    m_eq = np.array([m_1,m_2,m_3])
    x_vec = np.array(list(a)+list(m))

    return [A,m_eq,x_vec]


###########################################
#  - -> [L_1]
# |
# AT -> O_1 -> [O_2]  
###########################################
def model_4pr(a=np.empty(0),m=np.empty(0)): 
    
    global _bounds
    global _m_at_eq
    global _m_ocean_eq
    global _m_land_eq
    p = 4
    
    if(len(a)==0 and len(m)==0):
        a_size = 3
        m_size = 1
        
        info            = {}
        info['p']       = p
        info['name']    = '4PR' 
        info['m_names'] = ['AT','O_1','O_2','L_1'] 
    
        info['a_bounds'] = _bounds['a_level_1'] + _bounds['a_level_2'] + _bounds['a_level_1'] 
        info['m_bounds'] = _bounds['m_O1'] 
        
        return [a_size,m_size,info]
    
    ##################################
    # Assign variable
    ##################################
    
    A_21 = a[0] #AT-O_1
    A_32 = a[1] #O_1-O_2
    A_41 = a[2] #AT-L_1
    
    m_1 = _m_at_eq                      #AT
    m_2 =  m[0]                         #O_1
    m_3 = _m_ocean_eq - m_2             #O_3
    m_4 = _m_land_eq                    #L_1
    
    ##################################
    # Generate matrix
    ##################################
    A = np.zeros(shape=(p,p))
    
    A[1,0] = A_21
    A[2,1] = A_32
    A[3,0] = A_41
    
    A[0,1] = (A_21*m_1)/m_2
    A[1,2] = (A_32*m_2)/m_3
    A[0,3] = (A_41*m_1)/m_4

    temp = -np.sum(A,0)
    for i in range(0,p):
        A[i,i] = temp[i]
        
    m_eq = np.array([m_1,m_2,m_3,m_4])
    x_vec = np.array(list(a)+list(m))

    return [A,m_eq,x_vec]


###########################################
#  - -> L_1 - [L_2]
# |
# AT -> O_1 -> [O_2]  
###########################################
def model_5pr(a=np.empty(0),m=np.empty(0)): 
    
    global _bounds
    global _m_at_eq
    global _m_ocean_eq
    global _m_land_eq
    p = 5
    
    if(len(a)==0 and len(m)==0):
        a_size = 4
        m_size = 2
        
        info            = {}
        info['p']       = p
        info['name']    = '5PR' 
        info['m_names'] = ['AT','O_1','O_2','L_1','L_2'] 
    
        info['a_bounds'] = _bounds['a_level_1'] + _bounds['a_level_2'] + _bounds['a_level_1'] + _bounds['a_level_3'] 
        info['m_bounds'] = _bounds['m_O1'] + _bounds['m_L1'] 
        
        return [a_size,m_size,info]
    
    ##################################
    # Assign variable
    ##################################
    
    A_21 = a[0] #AT-O_1
    A_32 = a[1] #O_1-O_2
    A_41 = a[2] #AT-L_1
    A_54 = a[2] #L1-L_2
    
    m_1 = _m_at_eq                      #AT
    m_2 =  m[0]                         #O_1
    m_3 = _m_ocean_eq - m_2             #O_2
    m_4 =  m[1]                         #L_1
    m_5 = _m_land_eq  - m_4             #L_2
    
    ##################################
    # Generate matrix
    ##################################
    A = np.zeros(shape=(p,p))
    
    A[1,0] = A_21
    A[2,1] = A_32
    A[3,0] = A_41
    A[4,3] = A_54

    A[0,1] = (A_21*m_1)/m_2
    A[1,2] = (A_32*m_2)/m_3
    A[0,3] = (A_41*m_1)/m_4
    A[3,4] = (A_54*m_4)/m_5


    temp = -np.sum(A,0)
    for i in range(0,p):
        A[i,i] = temp[i]
        
    m_eq = np.array([m_1,m_2,m_3,m_4,m_5])
    x_vec = np.array(list(a)+list(m))

    return [A,m_eq,x_vec]



def model_test(a=np.empty(0),m=np.empty(0)): 

    global _bounds
    global _m_at_eq
    global _m_ocean_eq
    global _m_land_eq
    p = 4
    
    if(len(a)==0 and len(m)==0):
        a_size = 3
        m_size = 1
        
        info            = {}
        info['p']       = p
        info['name']    = '4PRD' 
        info['m_names'] = ['AT','O_1','O_2','L_1'] 
    
        info['a_bounds'] = _bounds['a_level_1'] + _bounds['a_level_2'] + _bounds['a_level_1'] 
        info['m_bounds'] = _bounds['m_O1'] 
        
        return [a_size,m_size,info]
    
    ##################################
    # Assign variable
    ##################################
    
    A_21 = 0.052 #AT-O_1
    A_32 = 0.0082 #O_1-O_2
    A_41 = 0.0125 #AT-L_1
    
    m_1 = 607                      #AT
    m_2 = 234                       #O_1
    m_3 = 1570              #O_3
    m_4 = 445                    #L_1
    
    ##################################
    # Generate matrix
    ##################################
    A = np.zeros(shape=(p,p))
    
    A[1,0] = A_21
    A[2,1] = A_32
    A[3,0] = A_41
    
    A[0,1] = (A_21*m_1)/m_2
    A[1,2] = (A_32*m_2)/m_3
    A[0,3] = (A_41*m_1)/m_4

    temp = -np.sum(A,0)
    for i in range(0,p):
        A[i,i] = temp[i]
        
    m_eq = np.array([m_1,m_2,m_3,m_4])
    x_vec = np.array(list(a)+list(m))

    return [A,m_eq,x_vec]


def check_error():
    
    model_set = [model_3sr,model_4pr,model_5pr,model_test]
    
    N=int(1e4)
    
    for model in model_set:
        
        temp=[0,0,0]
        [a_size,m_size,info]=model()
        t=0
            
        for i in range(0,N):
            
            a = np.random.rand(a_size)/2
            m = np.random.rand(m_size)*1000
            
            t += -time.time()
            [A,m_eq,x_vec]=model(a,m)
            t += time.time()



            #print('ttt=',np.array(list(a)+list(m)))
            #print('x_vec=',x_vec)
            
            temp[0] += A@m_eq/N
            temp[1] += np.ones(A.shape[0]).T@A/N
            temp[2] += (x_vec - np.array(list(a)+list(m)))/N
            
           
        if np.linalg.norm(temp[0])/len(temp[0])<1e-12 and np.linalg.norm(temp[1])/len(temp[1])<1e-12 and np.linalg.norm(temp[2])/len(temp[2])<1e-12:
            print(info['name'],'- OK [',np.round(t,2),'s for ',N,' calls]')
        else:
            
            print(info['name'],'- Error')
            print('-----------------------------------------------------------')
            print('a=',a)
            print('m=',m)
            print('A=',A)
            print('m_eq=',m_eq)
            print('A@m_eq=',temp[0])
            print('np.ones(A.shape[0]).T@A=',temp[1])
            print('x_vec - np.array(list(a)+list(m))=',temp[2])
            print('-----------------------------------------------------------')
        
  
if __name__ == "__main__":
    check_error()  
    
        
    
    
