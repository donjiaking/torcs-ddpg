import torch
from gym_torcs import TorcsEnv
import torch.nn as nn
from torch import optim
import numpy as np
import torch.nn.functional as F
import os

from buffer import ReplayBuffer
from actor import Actor
from critic import Critic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################## Hyperparameters ####################################

max_epoch = 10
max_time = 1000

dim_state = 22
dim_action = 3

batch_size = 32
buffer_capacity = 1000

vision = True

gamma = 0.9
tau = 0.001
epsilon = 1
epsilon_reduction = 1.0/100000

init_lr_actor = 0.0001
init_lr_critic = 0.001

checkpoint_freq = 5

is_train = True


############################## Initialization ####################################

actor = Actor(dim_state=dim_state).to(device)
critic = Critic(dim_state=dim_state, dim_action=dim_action).to(device)

target_actor = Actor(dim_state=dim_state).to(device)
target_critic = Critic(dim_state=dim_state, dim_action=dim_action).to(device)

if(os.path.exists('./models/actor_model.pth')):
    actor.load_state_dict(torch.load('./models/actor_model.pth'))
if(os.path.exists('./models/critic_model.pth')):
    critic.load_state_dict(torch.load('./models/critic_model.pth'))

replay_buffer = ReplayBuffer(buffer_capacity)

criterion_critic = nn.MSELoss(reduction='sum')

optimizer_actor = optim.Adam(actor.parameters(), lr=init_lr_actor)
optimizer_critic = optim.Adam(critic.parameters(), lr=init_lr_critic)

# generate a Torcs environment
env = TorcsEnv(vision=vision, throttle=False, gear_change=False)


############################### Util functions ####################################

def extract_state(ob):
    # TODO: add other sensors
    # may need to modify gym_torcs
    return np.hstack((ob.track, ob.speedX, ob.speedY, ob.speedZ))

def save_model(actor, critic):
    if(not os.path.exists('./models')):
        os.mkdir('./models')
    torch.save(actor.state_dict(), os.path.join('./models','actor_model.pth'))
    torch.save(critic.state_dict(), os.path.join('./models','critic_model.pth'))
    print("models saved")

def ou_process(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)

def generate_noise(a_t, eps):
    # generate noise by using Ornstein-Uhlenbeck process
    n_t = np.zeros([1, dim_action])
    n_t[0][0] = is_train * max(eps, 0) * ou_process(a_t[0][0], 0.0, 0.60, 0.30)
    n_t[0][1] = is_train * max(eps, 0) * ou_process(a_t[0][1], 0.5, 1.00, 0.10)
    n_t[0][2] = is_train * max(eps, 0) * ou_process(a_t[0][2], -0.1, 1.00, 0.05)
    return n_t

def load_one_batch():
    batch = replay_buffer.sample(batch_size=batch_size)
    s_i, a_i, r_i, s_i1 = [], [], [], []
    for tran in batch:
        s_i.append(tran[0])
        a_i.append(tran[1])
        r_i.append([tran[2]])
        s_i1.append(tran[3])
    s_i = torch.tensor(np.array(s_i).astype(np.float32), device=device)
    a_i = torch.tensor(np.array(a_i).astype(np.float32), device=device)
    r_i = torch.tensor(np.array(r_i).astype(np.float32), device=device)
    s_i1 = torch.tensor(np.array(s_i1).astype(np.float32), device=device)
    # each is 2-d tensor with size batch_size*feature_dimension
    return s_i, a_i, r_i, s_i1


################################# Train or Test #################################

print("Started!")
for e in range(max_epoch):
    print("Episode : " + str(e))

    if np.mod(e, 3) == 0:
        # deal with memory leak error
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()

    # initialize s_t as 1-d numpy array
    s_t = extract_state(ob)

    for t in range(max_time):
        epsilon -= epsilon_reduction

        a_t = actor(torch.stack([torch.tensor(s_t.astype(np.float32), device=device)],dim=0))
        a_t = a_t.data.cpu().numpy()
        n_t = generate_noise(a_t, epsilon)
        a_t = (a_t + n_t)[0]

        ob, reward, done, _ = env.step(a_t)
        r_t = reward
        s_t1 = extract_state(ob)

        replay_buffer.store(s_t, a_t, r_t, s_t1) # store transition in R

        s_i, a_i, r_i, s_i1 = load_one_batch() # sample batch from R
        y_i = torch.ones_like(r_i, device=device).float()

        target_q = target_critic(s_i1, target_actor(s_i1)) # B*1
        y_i = r_i + gamma * target_q # B*1

        if(is_train):
            # update critic network
            q = critic(s_i, a_i)
            optimizer_critic.zero_grad()
            loss = criterion_critic(y_i, q)
            loss.backward(retain_graph=True)
            optimizer_critic.step()

            # update actor network
            a = actor(s_i)
            q = critic(s_i, a)
            critic.zero_grad()
            q_sum = q.sum()
            grads = torch.autograd.grad(q_sum, a)

            a = actor(s_i)
            actor.zero_grad()
            a.backward(-grads[0])
            optimizer_actor.step()

            # update target network
            target_actor_state_dict = {}
            target_critic_state_dict = {}
            # print(target_actor.state_dict())

            for name in target_actor.state_dict():
                target_actor_state_dict[name] = tau*actor.state_dict()[name] 
                + (1-tau)*target_actor.state_dict()[name]
            target_actor.load_state_dict(target_actor_state_dict)

            for name in target_critic.state_dict():
                target_critic_state_dict[name] = tau*critic.state_dict()[name] 
                + (1-tau)*target_critic.state_dict()[name]
            target_critic.load_state_dict(target_critic_state_dict)

        # update current state
        s_t = s_t1
        
        # checkpoint
        if np.mod(e, checkpoint_freq) == 0 and is_train:
            save_model(actor, critic)
            


env.end()  # This is for shutting down TORCS
print("Finish!")
