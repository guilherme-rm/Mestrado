"""Resolve optional settings once and validate experiments before allocating resources."""
from dataclasses import dataclass
import math
import warnings
from functions.runtime_settings import apply_runtime_settings


def config_value(config, key, default):
    value = getattr(config, key, None)
    return default if value is None else value


@dataclass(frozen=True)
class ExperimentOptions:
    nepisodes: int
    nsteps: int
    nagents: int
    batch_size: int
    capacity: int
    learn_every: int = 1
    nupdate: int = 50
    gamma: float = .95

    def validate(self):
        for key in ('nepisodes', 'nsteps', 'nagents', 'batch_size', 'capacity', 'learn_every', 'nupdate'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError('gamma must be between 0 and 1')
        if self.batch_size > self.capacity:
            raise ValueError('batch_size cannot exceed replay capacity')


def resolve_configuration(opt, sce):
    defaults = dict(learn_every=1, nupdate=50, gamma=.95, step_log_throttle=10,
                    checkpoint_interval=0, momentum=0., min_memory_for_learning=0,
                    observation_mode='qos', architecture='mlp',
                    bootstrap_on_termination=True, target_update_clock='episode',
                    replay_sampling='legacy', gnn_encoder_training='fixed',
                    seed=None, mobility_enabled=False, mobility_decisions_per_move=0)
    for key, value in defaults.items():
        if getattr(opt, key, None) is None:
            setattr(opt, key, value)
    if getattr(opt, 'learning_rate', None) is None:
        opt.learning_rate = config_value(opt, 'learningrate', .0005)
    apply_runtime_settings(opt, sce)
    options = ExperimentOptions(**{key: getattr(opt, key) for key in ExperimentOptions.__dataclass_fields__})
    options.validate()
    for key in ('checkpoint_interval', 'min_memory_for_learning', 'mobility_decisions_per_move'):
        value = getattr(opt, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if not isinstance(opt.step_log_throttle, int) or opt.step_log_throttle < 1:
        raise ValueError('step_log_throttle must be a positive integer')
    if not math.isfinite(opt.learning_rate) or opt.learning_rate <= 0:
        raise ValueError('learning_rate must be positive and finite')
    for key, allowed in dict(observation_mode=('qos', 'enriched'), architecture=('mlp',),
                             replay_sampling=('legacy', 'with_replacement', 'without_replacement'),
                             target_update_clock=('episode', 'global'), gnn_encoder_training=('fixed',)).items():
        if getattr(opt, key) not in allowed:
            raise ValueError(f'{key} must be one of {allowed}')
    if opt.gnn_enabled:
        if opt.observation_mode == 'enriched':
            raise ValueError('Enriched flat observations cannot be combined with a graph encoder')
        if opt.gnn_observation_mode != 'replace' or config_value(opt, 'gnn_heterogeneous', False):
            raise ValueError('Only homogeneous graph observations in replace mode are implemented')
    if opt.environment_type not in ('hetnet', 'cell_free'):
        raise ValueError('Unsupported environment_type')
    if int(sce.nChannel) < 1 or float(sce.BW) <= 0:
        raise ValueError('nChannel and BW must be positive')
    if opt.environment_type == 'hetnet':
        if sce.nMBS < 1 or sce.nPBS < 0 or sce.nFBS < 0 or sce.nPBS > 4 * sce.nMBS:
            raise ValueError('HetNet requires MBS>=1, nonnegative small-cell counts, and PBS<=4*MBS')
        if any(float(getattr(sce, key)) <= 0 for key in ('rMBS', 'rPBS', 'rFBS')):
            raise ValueError('Coverage radii must be positive')
    if config_value(opt, 'parallel_agent_optimization', False):
        warnings.warn('parallel_agent_optimization is not implemented; optimization is sequential', UserWarning)
        opt.parallel_agent_optimization = False
    return opt
