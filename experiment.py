from logging import Logger
import torch
from torch.functional import Tensor
from gym_torcs import TorcsEnv
import torch.nn as nn
from torch import optim
import numpy as np
import torch.nn.functional as F
import os

from buffer import ReplayBuffer
from actor import Actor
from critic import Critic
from logger import get_logger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


############################## Hyperparameters ####################################

max_epoch = 1500
max_step = 100000

dim_state = 29
dim_action = 3

batch_size = 32
buffer_capacity = 100000

vision = False

gamma = 0.99
tau = 0.001
epsilon = 0.1 #1.5
epsilon_brake = 1.0 #1.5
epsilon_reduction = 1.0/100000
epsilon_brake_reduction = 10.0/100000

init_lr_actor = 0.0001
init_lr_critic = 0.001

checkpoint_freq = 10

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

target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

replay_buffer = ReplayBuffer(buffer_capacity)

# criterion_critic = nn.MSELoss(reduction='sum')

optimizer_actor = optim.Adam(actor.parameters(), lr=init_lr_actor)
optimizer_critic = optim.Adam(critic.parameters(), lr=init_lr_critic)

# generate a Torcs environment
env = TorcsEnv(vision=vision, throttle=True, gear_change=False)

logger = get_logger('log.txt')

############################### Util functions ####################################

def extract_state(ob):
    # TODO: add other sensors
    # Note: may need to modify gym_torcs.py to add some sensors
    # Note: some of them in ob are already normalized
    # Note: when change state, change dim_state as well
    return np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,
     ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

def save_model(actor, critic):
    if(not os.path.exists('./models')):
        os.mkdir('./models')
    torch.save(actor.state_dict(), os.path.join('./models','actor_model.pth'))
    torch.save(critic.state_dict(), os.path.join('./models','critic_model.pth'))
    print("models saved")

def ou_process(x, mu, theta, sigma):
    np.random.seed()
    return theta * (mu - x) + sigma * np.random.randn(1)

def generate_noise(a_t, eps):
    # generate noise by using Ornstein-Uhlenbeck process
    n_t = np.zeros([1, dim_action])
    n_t[0][0] = is_train * max(eps, 0) * ou_process(a_t[0][0], 0.0, 0.60, 0.30)
    n_t[0][1] = is_train * max(eps, 0) * ou_process(a_t[0][1], 0.5, 1.00, 0.10)
    n_t[0][2] = is_train * max(eps, 0) * ou_process(a_t[0][2], -0.1, 1.00, 0.05)

    # apply random brake
    if np.random.random() < 0.4: # TODO: tune this
        n_t[0][2] = is_train * max(epsilon_brake, 0) * ou_process(a_t[0][2], 0.2, 1.00, 0.10)

    return n_t

def load_one_batch():
    batch = replay_buffer.sample(batch_size=batch_size)
    s_i, a_i, r_i, s_i1, dones = [], [], [], [], []

    for tran in batch:
        s_i.append(tran[0])
        a_i.append(tran[1])
        r_i.append([tran[2]])
        s_i1.append(tran[3])
        dones.append([tran[4]])

    s_i = torch.tensor(np.array(s_i).astype(np.float32), device=device)
    a_i = torch.tensor(np.array(a_i).astype(np.float32), device=device)
    r_i = torch.tensor(np.array(r_i).astype(np.float32), device=device)
    s_i1 = torch.tensor(np.array(s_i1).astype(np.float32), device=device)
    dones = torch.tensor(np.array(dones), device=device)

    # each is 2-d tensor with size batch_size*feature_dimension
    return s_i, a_i, r_i, s_i1, dones


################################# Train or Test ##################################

logger.info("Started!!!!")
logger.info("------------------------------------------------")
for e in range(max_epoch):
    logger.info("Episode : " + str(e))

    if np.mod(e, 3) == 0:
        # deal with memory leak error
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()

    # initialize s_t as 1-d numpy array
    s_t = extract_state(ob)

    tot_reward = 0.
    tot_loss_critic = 0.
    tot_loss_actor = 0.
    tot_steps = 0

    for t in range(max_step):
        epsilon -= epsilon_reduction
        epsilon_brake_reduction -= epsilon_reduction

        a_t = actor(torch.stack([torch.tensor(s_t.astype(np.float32), device=device)],dim=0))
        a_t = a_t.data.cpu().numpy()
        n_t = generate_noise(a_t, epsilon)
        a_t = (a_t + n_t)[0]

        ob, reward, done, _ = env.step(a_t)
        r_t = reward
        s_t1 = extract_state(ob)
        tot_reward += r_t

        if(is_train):
            replay_buffer.store(s_t, a_t, r_t, s_t1, done) # store a transition in R

            s_i, a_i, r_i, s_i1, dones = load_one_batch() # sample a batch from R
            target_q = torch.ones_like(r_i, device=device).float()

            target_q_next = target_critic(s_i1, target_actor(s_i1)) # B*1
            target_q = r_i + gamma * target_q_next # B*1
            for m in range(batch_size):
                if(dones[m][0] == True):
                    target_q[m][0] = r_i[m][0]

            ## update critic network
            q = critic(s_i, a_i)
            optimizer_critic.zero_grad()
            loss_critic = F.mse_loss(target_q, q) # MSE as loss function
            loss_critic.backward()
            tot_loss_critic += loss_critic.item()
            optimizer_critic.step()

            ## update actor network
            # a = actor(s_i)
            # q = critic(s_i, a)
            # tot_loss_actor += -q.mean().item()
            # optimizer_critic.zero_grad()
            # q_sum = q.sum()
            # q_sum.backward(retain_graph=True)
            # grads = torch.autograd.grad(q_sum, a)

            # a = actor(s_i)
            # optimizer_actor.zero_grad()
            # a.backward(-grads[0])
            # optimizer_actor.step()

            a = actor(s_i)
            loss_actor = -critic(s_i, a).mean() # -Q as loss function
            tot_loss_actor += loss_actor.item()
            optimizer_actor.zero_grad()
            loss_actor.backward()
            optimizer_actor.step()

            ## update target network
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

        if(is_train):
            logger.info("Episode {} Step {}: Reward {:.3f} Critic Loss {:.6f}".format(e, t, r_t, loss_critic))
        else:
            logger.info("Episode {} Step {}: Reward {:.3f}".format(e, t, r_t))

        # update current state
        s_t = s_t1
        # update total steps that has been made
        tot_steps += 1

        # if done is True, terminate this episode
        if(done):
            break
        
    # checkpoint
    if np.mod(e, checkpoint_freq) == 0 and is_train:
        save_model(actor, critic)
    
    logger.info("TOTAL REWARD @ Episode {}: {:.3f}".format(e, tot_reward))
    logger.info("TOTAL STEPS: " + str(tot_steps))
    logger.info("TOTAL DISTANCE: " + str(ob.distRaced))
    if(is_train):
        logger.info("MEAN CRITIC LOSS: {:.6f}".format(tot_loss_critic/tot_steps))
        logger.info("MEAN ACTOR LOSS: {:.6f}".format(tot_loss_actor/tot_steps))
    logger.info("------------------------------------------------")



env.end()  # This is for shutting down TORCS
logger.info("Finish!")
