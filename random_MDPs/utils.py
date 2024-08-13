import numpy as np
import metrics
import sys

def get_normalized_matrix(N):
    T=np.random.random((N*N)).reshape(N,N)
    for i in range(N):
        T[i,:]=T[i,:]/np.sum(T[i,:])
    return T

def get_normalized_matrix_policy(N, n_actions):
    T = np.random.random((N * N * n_actions)).reshape(
        N,
        n_actions,
        N,
    )
    for i in range(N):
        for action in range(n_actions):
            T[i, action, :] = T[i, action, :] / np.sum(T[i, action, :])
    return T

def compute_planning_error(T,T_hat,R,gamma,N):
    I=np.identity(N)
    V=np.linalg.inv(I-gamma*T).dot(R)
    V_hat=np.linalg.inv(I-gamma*T_hat).dot(R)
    error=np.linalg.norm(V-V_hat,ord=2)
    return error

def compute_planning_error_policy(T, T_hat, R, gamma, N, policy):
    I = np.identity(N)
    T_pi = np.array([[T[i,policy[i],j].item() for j in range(N)] for i in range(N)])
    T_hat_pi = np.array(
        [[T_hat[i, policy[i], j].item() for j in range(N)] for i in range(N)]
    )
    V = np.linalg.inv(I - gamma * T_pi).dot(R)
    V_hat = np.linalg.inv(I - gamma * T_hat_pi).dot(R)
    error = np.linalg.norm(V - V_hat, ord=2)
    return error

def experiment(num_experiments,gamma,N,reward_type,policy=None,n_actions=1):
    li_TV,li_KL,li_W=[],[],[]
    li_planning_error=[]
    li_lip_T, li_lip_T_hat = [], []
    for experiment in range(num_experiments):
        np.random.seed(experiment) #choose a seed for reproducability
        if policy is not None:
            T,T_hat=get_normalized_matrix_policy(N, n_actions), get_normalized_matrix_policy(N, n_actions)
        else:
            T,T_hat=get_normalized_matrix(N),get_normalized_matrix(N)

        if reward_type=='random':
            R=np.random.random(N).reshape(N,1)
        elif reward_type=='structured':
            R=np.array([x for x in range(N)]).reshape(N,1)
        else:
            print("undefined reward structure ...")
            sys.exit(1)

        if policy is not None:
            policy = np.random.randint(low=0, high=n_actions, size=(N,1))
            planning_error = compute_planning_error_policy(T,T_hat,R,gamma,N,policy)
        else:
            planning_error = compute_planning_error(T,T_hat,R,gamma,N)

        li_planning_error.append(planning_error)
        (
            li_TV.append(metrics.TV(T.reshape((-1, N)), T_hat.reshape((-1, N)))),
            li_KL.append(metrics.KL(T.reshape((-1, N)), T_hat.reshape((-1, N)))),
            li_W.append(metrics.W(T.reshape((-1, N)), T_hat.reshape((-1, N)))),
        )

        # lipschitz cinstant of transition functions
        li_lip_T.append(metrics.Lip(T))
        li_lip_T_hat.append(metrics.Lip(T_hat))

    return li_TV, li_KL, li_W, li_planning_error, li_lip_T, li_lip_T_hat


def compute_covariance(li_1,li_2):
    return np.cov(li_1, li_2)[0,1]/np.sqrt(np.var(li_1)*np.var(li_2))