import abc
import datetime
import importlib
import os
import time
import json

import numpy as np
import torch
import tensorboardX

from infirun.util.schematic import Schematic
from infirun.util.reflect import get_source_data, ensure_basic_data_type


class EpochEnd(Exception):
    pass


class ModelComponent(Schematic):
    def __init__(self, cls):
        self.cls = cls

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self

    @staticmethod
    def _build_if_component(arg):
        return arg.build() if isinstance(arg, ModelComponent) else arg

    def build(self):
        args = [self._build_if_component(arg) for arg in self.args]
        kwargs = {k: self._build_if_component(arg) for k, arg in self.kwargs.items()}
        return self.cls(*args, **kwargs)

    def _get_schema(self):
        cls_info = get_source_data(self.cls)
        assert cls_info['type'] == 'cls'
        return {'mod': cls_info['mod'],
                'name': cls_info['name'],
                'args': [self._get_arg_schema(a) for a in self.args],
                'kwargs': {k: self._get_arg_schema(a) for k, a in self.kwargs.items()}}

    @staticmethod
    def _get_arg_schema(arg):
        if isinstance(arg, ModelComponent):
            return arg.get_schema()
        else:
            return ensure_basic_data_type(arg)

    @staticmethod
    def from_schema(schema):
        args = [ModelComponent._from_schema_if_component(a) for a in schema['args']]
        kwargs = {k: ModelComponent._from_schema_if_component(a) for k, a in schema['kwargs'].items()}
        mod = importlib.import_module(schema['mod'])
        cls = getattr(mod, schema['name'])
        return ModelComponent(cls)(*args, **kwargs)

    @staticmethod
    def _from_schema_if_component(schema):
        if schema['type'] == 'const':
            return schema['value']
        else:
            return ModelComponent.from_schema(schema)


class StatProcessor:
    state_keys = ('last_schedule_step_or_epoch',
                  'global_step',
                  'global_epoch',
                  'step_since_last_flush',
                  'last_validate_step',
                  'last_validate_epoch',
                  'last_save_step',
                  'last_save_time')

    def __init__(self,
                 log_per_n_steps,
                 save_per_n_steps,
                 save_per_minutes,
                 scheduler,
                 validate_per_n_steps,
                 validate_per_n_epochs,
                 schedule_per_n_steps,
                 schedule_per_n_epochs):
        self.scheduler = scheduler
        self.log_per_n_steps = log_per_n_steps
        self.save_per_n_steps = save_per_n_steps
        self.save_per_minutes = save_per_minutes

        self.validate_per_n_step = validate_per_n_steps
        self.validate_per_n_epochs = validate_per_n_epochs

        self.should_schedule_on_steps = schedule_per_n_steps is not None and schedule_per_n_steps > 0
        self.should_schedule_on_epochs = schedule_per_n_epochs is not None and schedule_per_n_epochs > 0
        assert not (self.should_schedule_on_steps and self.should_schedule_on_epochs)
        self.schedule_frequency = schedule_per_n_steps or schedule_per_n_epochs

        self.last_schedule_step_or_epoch = 0

        self.global_step = 0
        self.global_epoch = 0
        self.step_since_last_flush = 0

        self.last_validate_step = 0
        self.last_validate_epoch = 0

        self.last_save_step = 0
        self.last_save_time = time.time()
        self.last_stat = {}
        self.mean_stat = {}
        self.last_val_stat = {}

    def get_state(self):
        states = {k: self.__dict__[k] for k in self.state_keys}
        states['scheduler_state'] = self.scheduler.state_dict() if self.scheduler else None
        states['last_stat'] = self._serialize_stats(self.last_stat)
        states['mean_stat'] = self._serialize_stats(self.mean_stat)
        states['last_val_stat'] = self._serialize_stats(self.last_val_stat)
        return states

    def restore_state(self, state_dict):
        for k, v in state_dict.items():
            if k in self.state_keys:
                self.__dict__[k] = v
        if self.scheduler and ('scheduler_state' in state_dict):
            self.scheduler.load_state_dict(state_dict['scheduler_state'])

    def step(self, n, stats):
        stats = self._sanitize_values(stats)
        self.global_step += n
        self.step_since_last_flush += n
        self.last_stat = stats
        for k, v in stats.items():
            if self._is_mean_key(k):
                try:
                    self.mean_stat[k] += v
                except KeyError:
                    self.mean_stat[k] = v
            else:
                self.last_stat[k] = v

    def step_epoch(self, n=1):
        self.global_epoch += n

    @staticmethod
    def _sanitize_values(stats):
        return {k: StatProcessor._sanitize_value(v) for k, v in stats.items()}

    @staticmethod
    def _sanitize_value(stat):
        if isinstance(stat, torch.Tensor):
            return stat.detach().cpu().numpy()
        else:
            return stat

    @staticmethod
    def _serialize_stats(state_dict):
        ret = {}
        for k, v in state_dict.items():
            if np.isscalar(v):
                ret[k] = v
            try:
                if v.shape == ():
                    ret[k] = np.asscalar(v)
            except AttributeError:
                raise
        return ret

    @staticmethod
    def _is_mean_key(k):
        splitted = k.split('.')
        return splitted[-1].startswith('mean_')

    def flush_stat_if_should_log(self, log_writer):
        if self.step_since_last_flush >= self.log_per_n_steps:
            for k, v in self.last_stat.items():
                self._do_write_log(log_writer, k, v, self.global_step)
            for k, v in self.mean_stat.items():
                self._do_write_log(log_writer, k, v / self.step_since_last_flush, self.global_step)
            self.step_since_last_flush = 0
            self.mean_stat.clear()
            self.last_stat.clear()

    def update_status_if_should_save(self):
        now = time.time()
        should_save_by_steps = self._should_save_by_steps()
        should_save_by_time = self._should_save_by_time(now)
        if (should_save_by_steps is None and should_save_by_time is None) \
                or should_save_by_steps is False or should_save_by_time is False:
            return False
        self.last_save_step = self.global_step
        self.last_save_time = now
        return True

    def _should_save_by_steps(self):
        if self.save_per_n_steps is None:
            return None
        return self.global_step - self.last_save_step >= self.save_per_n_steps

    def _should_save_by_time(self, now):
        if self.save_per_minutes is None:
            return None
        return now - self.last_save_time >= self.save_per_minutes * 60

    def should_validate(self):
        should_validate_by_steps = self._should_validate_by_steps()
        should_validate_by_epochs = self._should_validate_by_epochs()
        if (should_validate_by_epochs is None and should_validate_by_steps is None) \
                or should_validate_by_epochs is False or should_validate_by_steps is False:
            return False
        return True

    def step_validate(self, log_writer, stats):
        self.last_validate_step = self.global_step
        self.last_validate_epoch = self.global_epoch
        stats = self._sanitize_values(stats)
        for k, v in stats.items():
            self.last_val_stat[k] = v
            self._do_write_log(log_writer, k, v, self.global_step)

    def _should_validate_by_steps(self):
        if self.validate_per_n_step is None:
            return None
        return self.global_step - self.last_validate_step >= self.validate_per_n_step

    def _should_validate_by_epochs(self):
        if self.validate_per_n_epochs is None:
            return None
        return self.global_epoch - self.last_validate_epoch >= self.validate_per_n_epochs

    def step_scheduler(self):
        if not (self.scheduler or self.should_schedule_on_steps or self.should_schedule_on_epochs):
            return
        current_step_or_epoch = self.global_step if self.should_schedule_on_steps else self.global_epoch
        if (current_step_or_epoch - self.last_schedule_step_or_epoch) < self.schedule_frequency:
            return

        self.scheduler.step(current_step_or_epoch)
        self.last_schedule_step_or_epoch = current_step_or_epoch

    @staticmethod
    def _do_write_log(log_writer, k, v, step):
        if log_writer is None:
            return print(f'Step {step} {k} -- {v}')
        try:
            val_type = v['type']
            value = v['value']
            del v['type']
            del v['value']
            getattr(log_writer, 'add_' + val_type)(k, value, global_step=step, **v)
        except IndexError:
            log_writer.add_scalar(k, v, global_step=step)


class Trainer(abc.ABC):
    """
    what can this shite do?

    * automatic construction of sources and model ... OK, sorta
    * step through training ... OK
    * return maps that contain stats at each step ... OK
    * setup and teardown (mainly setup, where things like initializing optimizers are done) ... OK
    * validation and logging are automatically done based on setting ... OK
    * steps are measured irrespective to batch size ... OK
    * stepping signals: measuring epochs ... OK
    * automatic saving and restoring of models ... OK
    * automatic stopping and resuming ... None of my business
    * restoring from other model ... Later
    """

    def __init__(self,
                 model,
                 train_source,
                 validate_source,
                 log_per_n_steps,
                 save_per_n_steps,
                 save_per_minutes,
                 keep_last_n_saves,
                 validate_per_n_steps,
                 validate_per_n_epochs,
                 validate_steps,
                 schedule_per_n_steps,
                 schedule_per_n_epochs,
                 model_dir,
                 log_dir,
                 actual_resources=None,
                 other_runs=None,
                 settings=None):
        self.settings = settings or {}
        self.model = model.build()
        self._load_state_from_other_runs(other_runs)
        self.train_model = self.prepare_model_for_train(self.model,
                                                        self.declare_compute_resource(),
                                                        actual_resources or {},
                                                        self.settings)
        self.optimizer = self.setup_optimizer(self.train_model, self.settings)
        self.scheduler = self.setup_scheduler(self.optimizer, self.settings)
        self.train_source = train_source
        self.validate_source = validate_source
        self.stat_proc = StatProcessor(scheduler=self.scheduler,
                                       log_per_n_steps=log_per_n_steps,
                                       save_per_n_steps=save_per_n_steps,
                                       save_per_minutes=save_per_minutes,
                                       validate_per_n_epochs=validate_per_n_epochs,
                                       validate_per_n_steps=validate_per_n_steps,
                                       schedule_per_n_epochs=schedule_per_n_epochs,
                                       schedule_per_n_steps=schedule_per_n_steps)
        self.model_dir = self._ensure_dir(model_dir)
        self.log_writer = log_dir and tensorboardX.SummaryWriter(log_dir=log_dir)
        self.keep_last_n_saves = keep_last_n_saves
        self.validate_steps = validate_steps

        if self._has_previous_model_data():
            self.restore_model_and_state()
        self.settings = {}

    @staticmethod
    def _ensure_dir(path):
        if path is None:
            return
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
        return path

    def _load_state_from_other_runs(self, state_dicts):
        if state_dicts:
            self.load_state_from_other_runs(self.model, state_dicts)

    def load_state_from_other_runs(self, model, state_dicts):
        pass

    # @abc.abstractmethod
    def declare_source_dependencies(self):
        return []

    # @abc.abstractmethod
    def declare_compute_resource(self):
        return {}

    # @abc.abstractmethod
    def prepare_model_for_train(self, model, requested_resources, actual_resources, settings):
        if 'GPU' in requested_resources:
            assert len(actual_resources['GPU']) == requested_resources['GPU'], 'not enough GPU allocated'
            return torch.nn.DataParallel(model, device_ids=actual_resources['GPU'])
        return model

    def get_model_state(self):
        return self.model.state_dict()

    def load_model_state(self, model, state_dict):
        model.load_state_dict(state_dict)

    # @abc.abstractmethod
    def _save_model_and_state(self, model, path):
        if self.model_dir is None:
            print('Not saving model because model_dir not specified:', path)
            return
        full_path = os.path.join(self.model_dir, path)
        full_path_pt = full_path + '.pt'
        full_path_json = full_path + '.json'
        latest_path = os.path.join(self.model_dir, 'latest')
        latest_path_pt = latest_path + '.pt'
        latest_path_json = latest_path + '.json'
        torch.save(self.get_model_state(), full_path_pt)
        with open(full_path_json, 'w', encoding='utf-8') as f:
            json.dump(self.stat_proc.get_state(), f, ensure_ascii=True, indent=2)
        try:
            os.remove(latest_path_pt)
            os.remove(latest_path_json)
        except FileNotFoundError:
            pass
        os.link(full_path_pt, latest_path_pt)
        os.link(full_path_json, latest_path_json)
        if self.keep_last_n_saves is not None and self.keep_last_n_saves > 0:
            self._cleanup_saves('.json')
            self._cleanup_saves('.pt')

    def _cleanup_saves(self, ext):
        save_files = [os.path.join(self.model_dir, file) for file in os.listdir(self.model_dir) if
                      file.endswith(ext) and file != 'latest' + ext]
        save_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        for f in save_files[self.keep_last_n_saves:]:
            os.remove(f)

    def _has_previous_model_data(self):
        return (os.path.exists(os.path.join(self.model_dir, 'latest.pt')) or os.path.exists(
            os.path.join(self.model_dir, 'latest.json')))

    # @abc.abstractmethod
    def restore_model_and_state(self, model_path=None, checkpoint_name='latest'):
        if model_path is None:
            model_path = self.model_dir
        pt_file = os.path.join(model_path, checkpoint_name + '.pt')
        json_file = os.path.join(model_path, checkpoint_name + '.json')
        self.load_model_state(self.model, torch.load(pt_file))
        with open(json_file, 'r', encoding='utf-8') as f:
            self.stat_proc.restore_state(json.load(f))

    @abc.abstractmethod
    def setup_optimizer(self, model, settings):
        """

        :param model:
        :return: the prepared optimizer
        """

    # @abc.abstractmethod
    def setup_scheduler(self, optimizer, settings):
        """

        :return: the prepared scheduler
        """
        return

    @abc.abstractmethod
    def train_step(self, model, next_values, settings):
        """

        :return: loss, {"mean_loss": xx, "mean_acc": xx}
        """

    # @abc.abstractmethod
    def optimize_step(self, loss, optimizer, settings):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    @abc.abstractmethod
    def validate(self, model, next_values, settings):
        """

        :return: {"validate_result": single, "another": [1,2,3,4,5]}
        """

    @staticmethod
    def _determine_n_samples(train_value):
        if isinstance(train_value, torch.Tensor):
            return train_value.shape[0]
        else:
            try:
                for k, v in train_value.items():
                    try:
                        return v.shape[0]
                    except AttributeError:
                        continue
                    pass
            except AttributeError:
                pass
            for v in train_value:
                try:
                    return v.shape[0]
                except AttributeError:
                    continue
        raise ValueError('cannot determine n_samples')

    def start(self):
        try:
            while True:
                try:
                    next_train_value = self.train_source.get()
                    n_samples = self._determine_n_samples(next_train_value)
                    loss, stats = self.train_step(self.train_model, next_train_value, self.settings)
                    self.optimize_step(loss, self.optimizer, self.settings)
                    self.stat_proc.step(n_samples, stats)
                    self._log_if_ncessary()
                    self._validate_if_necessary()
                    self._save_if_necessary()
                    self._step_scheduler_if_necessary()
                except EpochEnd:
                    self.stat_proc.step_epoch()
                    self._step_scheduler_if_necessary()
        except StopIteration:
            self._flush_writer()

    def _log_if_ncessary(self):
        self.stat_proc.flush_stat_if_should_log(self.log_writer)

    def _validate_if_necessary(self):
        if self.stat_proc.should_validate():
            validate_values = [self.validate_source.get() for _ in range(self.validate_steps)]
            validate_stats = self.validate(self.model, validate_values, self.settings)
            self.stat_proc.step_validate(self.log_writer, validate_stats)

    def _save_if_necessary(self):
        if self.stat_proc.update_status_if_should_save():
            time_str = datetime.datetime.now().isoformat('_', 'minutes')
            time_str = time_str.replace('-', '_').replace(':', '_').replace('.', '_')
            self._save_model_and_state(self.model,
                                       f'{time_str}_{self.stat_proc.global_epoch}_{self.stat_proc.global_step}')

    def _flush_writer(self):
        if self.log_writer is not None:
            self.log_writer.file_writer.flush()

    def _step_scheduler_if_necessary(self):
        if self.scheduler:
            self.stat_proc.step_scheduler()
