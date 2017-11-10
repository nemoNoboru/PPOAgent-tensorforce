from tensorforce import Configuration
from tensorforce.agents import PPOAgent
from tensorforce.core.preprocessing import Normalize
from server import Gym

print "Arrancando...."


def quadratic(l):
    return map(lambda x: abs(x), l)


config = Configuration(
    batch_size=1000,
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-4
    ),
    optimization_steps=5,
    likelihood_ratio_clipping=0.2,
    discount=0.99,
    normalize_rewards=True,
    entropy_regularization=1e-2,
    saver_spec=dict(directory='./agent', seconds=100),
    preprocessing=Normalize
)

# Create a Proximal Policy Optimization agent
agent = PPOAgent(
    states_spec=dict(type='float', shape=(4,)),
    actions_spec=dict(type='float', shape=(2)),
    network_spec=[
        dict(type='dense', size=32),
        dict(type='dense', size=32),
        #dict(type='dense', size=128),
        #dict(type='dense', size=128),
        #dict(type='dense', size=128),
        #dict(type='dense', size=128),
        #dict(type='dense', size=128),
        #dict(type='dense', size=128),
        #dict(type='dense', size=128)
    ],
    config=config
)
# agent.load_model('./saved_agent')
# Get new data from somewhere, e.g. a client to a web app
gym = Gym()
print "Listo!"

# Poll new state from client
state = gym.step(0)['state']
reward = 0
gens = 0
steps = 0
while True:
    # Get prediction from agent, execute
    # print state
    action = agent.act(state)
    observation = gym.step(action.tolist())
    # Add experience, agent automatically updates model according to batch size
    agent.observe(reward=observation['reward'], terminal=observation['done'])
    state = observation['state']

    # Statistics
    reward += observation['reward']
