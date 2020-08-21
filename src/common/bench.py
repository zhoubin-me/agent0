import atari_py
import gym

_atari = ["".join(list(map(lambda x: x.capitalize(), game.split('_')))) for game in atari_py.list_games()]
_bullet = [x.id[:-12] for x in gym.envs.registry.all() if 'BulletEnv' in x.id]

_atari8 = ['BeamRider', 'Breakout', 'Enduro', 'Pong', 'Qbert', 'Seaquest', 'SpaceInvaders', 'Asterix']
_atariexpl7 = ['Freeway', 'Gravitar', 'MontezumaRevenge', 'Pitfall', 'PrivateEye', 'Solaris', 'Venture']
_atari10 = ['BeamRider', 'Breakout', 'Enduro', 'Pong', 'Qbert', 'Seaquest', 'SpaceInvaders', 'Frostbite', 'MsPacman',
            'KungFuMaster']
_atari50 = [  # actually 47
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

_mujoco7 = [
    'Reacher', 'Hopper', 'HalfCheetah', 'Walker2D', "Ant", "Pusher", "Humanoid"
]
