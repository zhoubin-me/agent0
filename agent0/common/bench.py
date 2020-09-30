import atari_py
import gym

atari7 = ['Asterix', 'BeamRider', 'Breakout', 'Enduro', 'Qbert', 'Seaquest', 'SpaceInvaders']

atari_exp7 = ['Freeway', 'Gravitar', 'MontezumaRevenge', 'Pitfall', 'PrivateEye', 'Solaris', 'Venture']

atari11 = atari7 + ['MsPacman', 'Frostbite', 'Freeway', 'MontezumaRevenge']

atari47 = [  # actually 47
    'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids',
    'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider', 'Bowling',
    'Breakout', 'Centipede', 'ChopperCommand', 'CrazyClimber',
    'DemonAttack', 'DoubleDunk', 'Enduro', 'FishingDerby', 'Freeway',
    'Frostbite', 'Gopher', 'Gravitar', 'IceHockey', 'Jamesbond',
    'Kangaroo', 'Krull', 'KungFuMaster', 'MontezumaRevenge', 'MsPacman',
    'NameThisGame', 'Pitfall', 'Pong', 'PrivateEye', 'Qbert',
    'RoadRunner', 'Robotank', 'Seaquest', 'SpaceInvaders', 'StarGunner',
    'Tennis', 'TimePilot', 'Tutankham', 'UpNDown', 'Venture',
    'VideoPinball', 'WizardOfWor', 'Zaxxon',
]

atari63 = ["".join(list(map(lambda x: x.capitalize(), game.split('_')))) for game in atari_py.list_games()]

mujoco7 = ['Reacher', 'Hopper', 'HalfCheetah', 'Walker2D', "Ant", "Pusher", "Humanoid"]
bullet = [x.id[:-14] for x in gym.envs.registry.all() if 'BulletEnv' in x.id]
