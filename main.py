#coding: utf-8


import sys
import uuid

import numpy as np
import six
import tensorflow as tf
# import common
# from common import (play_model, Evaluator, eval_model_multithread,
#                     play_one_episode, play_n_episodes)
from autodrive.utils import logger
from simulator import *
from six.moves import queue

if six.PY3:
    from concurrent import futures
    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

GAMMA = 0.99
LOCAL_TIME_MAX = 5
STEPS_PER_EPOCH = 256
EVAL_EPISODE = 0
BATCH_SIZE = 64
PREDICT_BATCH_SIZE = 32     # batch for efficient forward
SIMULATOR_PROC = 5      #改成了cpu的个数
PREDICTOR_THREAD_PER_GPU = 2
PREDICTOR_THREAD = None
INIT_LEARNING_RATE_A = 1e-4
INIT_LEARNING_RATE_C = 2e-4
# EVALUATE_PROC = min(multiprocessing.cpu_count() // 2, 20)
CLIP_PARAMETER=0.2
NUM_ACTIONS = None

from autodrive.agent.torcs import AgentTorcs
from autodrive.agent.fake import AgentFake
clsAgent = AgentTorcs
# clsAgent = AgentFake
def get_player(agentIdent, viz=False, train=False, dumpdir=None):
    # 由于单机运行torcs实例有限，为了防止某些情况下所有agent产生样本相关性太大，此处加入同样数目的
    # 回放agent，随机播放由训练过程中torcs产生最佳3组episode纪录的
    if agentIdent < SIMULATOR_PROC:
        pl = clsAgent(agentIdent, is_train=train,
                      save_dir = '/tmp/torcs/memory/{:04d}'.format(agentIdent),
                      min_save_score = 50.,
                      max_save_item = 3,
                      )
    else:
        from autodrive.agent.base import AgentMemoryReplay
        pl = AgentMemoryReplay(agentIdent, is_train=train,
                               load_dir = '/tmp/torcs/memory/{:04d}'.format(agentIdent-SIMULATOR_PROC),
                               max_save_item = 3,
                               )
    return pl


from simulator import SimulatorProcess
class MySimulatorWorker(SimulatorProcess):
    def _build_player(self):
        log_dir =  '/tmp/torcs_run/'
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        log_file = log_dir + '/agent-{:03d}.log'.format(self.idx)
        sys.stdout = open(log_file, mode='w+')
        from autodrive.utils import logger, redict_logger_output
        redict_logger_output(sys.stdout)
        logger.info("agent {} start".format(self.idx))
        return get_player(agentIdent=self.idx, train=True)

from tensorpack.graph_builder.model_desc import ModelDesc, InputDesc
from tensorpack.tfutils.tower import get_current_tower_context
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.tfutils import summary
from tensorpack.tfutils.gradproc import SummaryGradient, GlobalNormClip, MapGradient
from tensorpack.tfutils import optimizer
class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None,6), 'state'),
                InputDesc(tf.float32, (None,2), 'action'),
                InputDesc(tf.float32, (None,), 'futurereward'),
                InputDesc(tf.float32, (None,), 'advantage'),
                # InputDesc(tf.float32, (None,), 'action_prob'),
                ]

    def _get_NN_prediction(self, state):
        from tensorpack.tfutils import symbolic_functions
        ctx = get_current_tower_context()
        is_training = ctx.is_training
        l = state
        # l = tf.Print(l, [state], 'State = ')
        with tf.variable_scope('critic') as vs:

            from autodrive.model.selu import fc_selu
            for lidx in range(8):
                l = fc_selu(l, 200,
                            keep_prob=1., # 由于我们只使用传感器训练，关键信息不能丢
                            is_training=is_training, name='fc-{}'.format(lidx))
            # l = tf.layers.dense(l, 512, activation=tf.nn.relu, name='fc-dense')
            # for lidx, hidden_size in enumerate([300, 600]):
            #     l = tf.layers.dense(l, hidden_size, activation=tf.nn.relu, name='fc-%d'%lidx)
            value = tf.layers.dense(l, 1, name='fc-value',\
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            if not hasattr(self, '_weights_critic'):
                self._weights_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope('actor') as vs:
            l = tf.stop_gradient(l)
            mu_steering = 0.5 * tf.layers.dense(l, 1, activation=tf.nn.tanh, name='fc-mu-steering',\
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            mu_accel = tf.layers.dense(l, 1, activation=tf.nn.tanh, name='fc-mu-accel',\
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            mus = tf.concat([mu_steering, mu_accel], axis=-1)
            # mus = tf.layers.dense(l, 2, activation=tf.nn.tanh, name='fc-mus')
            # sigmas = tf.layers.dense(l, 2, activation=tf.nn.softplus, name='fc-sigmas')
            # sigmas = tf.clip_by_value(sigmas, -0.001, 0.5)
            sigma_steering_ = 0.5 * tf.layers.dense(l, 1, activation=tf.nn.sigmoid, name='fc-sigma-steering',\
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            sigma_accel_ = 1. * tf.layers.dense(l, 1, activation=tf.nn.sigmoid, name='fc-sigma-accel',\
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            # sigma_beta_steering = symbolic_functions.get_scalar_var('sigma_beta_steering', 0.3, summary=True, trainable=False)
            # sigma_beta_accel = symbolic_functions.get_scalar_var('sigma_beta_accel', 0.3, summary=True, trainable=False)
            from tensorpack.tfutils.common import get_global_step_var
            sigma_beta_steering_exp = tf.train.exponential_decay(0.001, get_global_step_var(), 1000, 0.5, name='sigma/beta/steering/exp')
            sigma_beta_accel_exp = tf.train.exponential_decay(0.5, get_global_step_var(), 5000, 0.5, name='sigma/beta/accel/exp')
            # sigma_steering = tf.minimum(sigma_steering_ + sigma_beta_steering, 0.5)
            # sigma_accel = tf.minimum(sigma_accel_ + sigma_beta_accel, 0.2)
            # sigma_steering = sigma_steering_
            sigma_steering = (sigma_steering_ + sigma_beta_steering_exp)
            sigma_accel = (sigma_accel_ + sigma_beta_accel_exp) #* 0.1
            # sigma_steering = sigma_steering_
            # sigma_accel = sigma_accel_
            sigmas = tf.concat([sigma_steering, sigma_accel], axis=-1)
            #     sigma_steering = tf.clip_by_value(sigma_steering, 0.1, 0.5)

            #     sigma_accel = tf.clip_by_value(sigma_accel, 0.1, 0.5)

            # sigmas = sigmas_orig + 0.001
            # sigmas = tf.clip_by_value(sigmas, 0.1, 0.5)
            # sigma_beta = tf.get_variable('sigma_beta', shape=[], dtype=tf.float32,
            #                              initializer=tf.constant_initializer(.5), trainable=False)

            # if is_training:
            #     pass
            #     # 如果不加sigma_beta，收敛会很慢，并且不稳定，猜测可能是以下原因：
            #     #   1、训练前期尽量大的探索可以避免网络陷入局部最优
            #     #   2、前期过小的sigma会使normal_dist的log_prob过大，导致梯度更新过大，网络一开始就畸形了，很难恢复回来
            #
            # if is_training:
            #     sigmas += sigma_beta_steering
            # sigma_steering = tf.clip_by_value(sigma_steering, sigma_beta_steering, 0.5)
            # sigma_accel = tf.clip_by_value(sigma_accel, sigma_beta_accel, 0.5)
            # sigmas = tf.clip_by_value(sigmas, 0.1, 0.5)
            # sigmas_orig = sigmas
            # sigmas = sigmas + sigma_beta_steering
            # sigmas = tf.minimum(sigmas + 0.1, 100)
            # sigmas = tf.clip_by_value(sigmas, sigma_beta_steering, 1)
            # sigma_steering += sigma_beta_steering
            # sigma_accel += sigma_beta_accel

            # mus = tf.concat([mu_steering, mu_accel], axis=-1)

            from tensorflow.contrib.distributions import Normal
            dists = Normal(mus, sigmas+1e-3)
            actions = tf.squeeze(dists.sample([1]), [0])
            # 裁剪到一倍方差之内
            # actions = tf.clip_by_value(actions, -1., 1.)
            if is_training:
                summary.add_moving_summary(tf.reduce_mean(mu_steering, name='mu/steering/mean'),
                                           tf.reduce_mean(mu_accel, name='mu/accel/mean'),
                                           tf.reduce_mean(sigma_steering, name='sigma/steering/mean'),
                                           tf.reduce_max(sigma_steering, name='sigma/steering/max'),
                                           tf.reduce_mean(sigma_accel, name='sigma/accel/mean'),
                                           tf.reduce_max(sigma_accel, name='sigma/accel/max'),
                                           sigma_beta_accel_exp,
                                           sigma_beta_steering_exp,
                                           )
            # actions = tf.Print(actions, [mus, sigmas, tf.concat([sigma_steering_, sigma_accel_], -1), actions],
            #                    'mu/sigma/sigma.orig/act=', summarize=4)
            if not hasattr(self, '_weights_actor'):
                self._weights_actor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return actions, value, dists

    def _build_graph(self, inputs):
        from tensorpack.tfutils.common import get_global_step_var
        state, action, futurereward, advantage = inputs
        is_training = get_current_tower_context().is_training
        policy, value, dists = self._get_NN_prediction(state)
        if not hasattr(self, '_weights_train'):
            self._weights_train = self._weights_critic + self._weights_actor
        self.value = tf.squeeze(value, [1], name='value')  # (B,)
        self.policy = tf.identity(policy, name='policy')

        with tf.variable_scope("Pred") as vs:
            __p, __v, _ = self._get_NN_prediction(state)
            __v = tf.squeeze(__v, [1], name='value')  # (B,)
            __p = tf.identity(__p, name='policy')
            if not hasattr(self, '_weights_pred'):
                self._weights_pred = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
                assert (len(self._weights_train) == len(self._weights_pred))
                assert (not hasattr(self, '_sync_op'))
                self._sync_op = tf.group(*[d.assign(s + tf.truncated_normal(tf.shape(s), stddev=0.02)) for d, s in zip(self._weights_pred, self._weights_train)])

        with tf.variable_scope('pre') as vs:
            pre_p,pre_v,pre_dists=self._get_NN_prediction(state)
            if not hasattr(self,'pre_weights'):
                self.pre_weights=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=vs.name)
                self._td_sync_op = tf.group(*[d.assign(s) for d, s in zip(self.pre_weights, self._weights_train)])


        if not is_training:
            return

        # advantage = tf.subtract(tf.stop_gradient(self.value), futurereward, name='advantage')
        # advantage = tf.Print(advantage, [self.value, futurereward, action, advantage], 'value/reward/act/advantage=', summarize=4)
        log_probs = dists.log_prob(action)
        #add  ppo policy clip loss
        #add ratio  ,surr1, surr2
        pre_probs=pre_dists.log_prob(action)
        ratio=tf.exp(log_probs-pre_probs)
        prob_ratio = tf.reduce_mean(input_tensor=tf.concat(values=ratio, axis=1), axis=1)
        clip_param=tf.train.exponential_decay(CLIP_PARAMETER, get_global_step_var(), 10000, 0.98, name='clip_param')


        # surr1=prob_ratio*advantage
        surr1=ratio*tf.expand_dims(advantage, -1)
        surr2=tf.clip_by_value(ratio,1.0-clip_param,1.0+clip_param)*tf.expand_dims(advantage, -1)
        
        # surr2=tf.clip_by_value(prob_ratio,1.0-clip_param,1.0+clip_param)*advantage

        loss_policy=-tf.reduce_mean(tf.minimum(surr1,surr2))

        #add critic clip loss
        v_loss1=tf.square(value-futurereward)
        pre_value=pre_v+tf.clip_by_value(value-pre_v,-clip_param,clip_param)
        v_loss2=tf.square(pre_v-futurereward)
        # loss_value=0.5*tf.reduce_mean(tf.maximum(v_loss1,v_loss2))
        loss_value=0.5*tf.reduce_mean(v_loss1)
        

        entropy = dists.entropy()
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        exp_v = entropy_beta * entropy
        loss_entropy = tf.reduce_mean(-tf.reduce_sum(exp_v, axis=-1), name='loss/policy')
        loss_policy=loss_policy+loss_entropy
        

        # exp_v = tf.transpose(
        #     tf.multiply(tf.transpose(log_probs), advantage))
        # exp_v = tf.multiply(log_probs, advantage)
        # exp_v = log_probs * tf.expand_dims(advantage, -1)
        # entropy = dists.entropy()
        # entropy_beta = tf.get_variable('entropy_beta', shape=[],
        #                                initializer=tf.constant_initializer(0.01), trainable=False)
        # exp_v = entropy_beta * entropy + exp_v
        
        # loss_value = tf.reduce_mean(0.5 * tf.square(self.value - futurereward))

        # loss_entropy = tf.reduce_mean(tf.reduce_sum(entropy, axis=-1), name='xentropy_loss')


        from tensorflow.contrib.layers.python.layers.regularizers import apply_regularization, l2_regularizer
        loss_l2_regularizer = apply_regularization(l2_regularizer(1e-4), self._weights_critic)
        loss_l2_regularizer = tf.identity(loss_l2_regularizer, 'loss/l2reg')
        loss_value += loss_l2_regularizer
        loss_value = tf.identity(loss_value, name='loss/value')

        # self.cost = tf.add_n([loss_policy, loss_value * 0.1, loss_l2_regularizer])
        self._cost = [loss_policy,
                      loss_value
                      ]
        from autodrive.trainer.summary import addParamSummary
        addParamSummary([('.*', ['rms', 'absmax'])])
        pred_reward = tf.reduce_mean(self.value, name='predict_reward')
        advantage = symbf.rms(advantage, name='rms_advantage')
        summary.add_moving_summary(loss_policy, loss_value,
                                   loss_entropy,
                                   pred_reward, advantage,
                                   loss_l2_regularizer,
                                   tf.reduce_mean(self.policy[:, 0], name='action/steering/mean'),
                                   tf.reduce_mean(self.policy[:, 1], name='action/accel/mean'),
                                    )

    def _get_cost_and_grad(self):
        from tensorpack.tfutils.gradproc import FilterNoneGrad
        ctx = get_current_tower_context()
        assert ctx is not None and ctx.is_training, ctx

        # cost = self.get_cost()    # assume single cost
        loss_policy, loss_value = self._cost
        opt_a, opt_v = self.get_optimizer()
        grads_a = opt_a.compute_gradients(loss_policy, var_list=self._weights_actor, colocate_gradients_with_ops=True)
        grads_a = FilterNoneGrad().process(grads_a)
        grads_v = opt_v.compute_gradients(loss_value, var_list=self._weights_critic, colocate_gradients_with_ops=True)
        grads_v = FilterNoneGrad().process(grads_v)
        return self._cost, [grads_a, grads_v]
    #     # produce gradients
        # varlist = ctx.filter_vars_by_vs_name(tf.trainable_variables())
        # opt = self.get_optimizer()
        # grads = opt.compute_gradients(
        #     cost, var_list=varlist,
        #     gate_gradients=False, colocate_gradients_with_ops=True)
        #
        # return cost, grads

    def _calc_learning_rate(self, name, epoch, lr):
        def _calc():
            lr_init = INIT_LEARNING_RATE_A if name == 'actor' else INIT_LEARNING_RATE_C
            lrs = [(0, lr_init * 0.20),
                   (1, lr_init * 0.25),
                   (2, lr_init * 0.68),
                   (3, lr_init * 0.5),
                   (4, lr_init * 0.25),
                   (5, lr_init * 0.128),
                   # (100, lr_init/16),
                   ]
            for idx in range(len(lrs) - 1):
                if epoch >= lrs[idx][0] and epoch < lrs[idx+1][0]:
                    return lrs[idx][1]
            return lrs[-1][1]
        ret = _calc()
        return ret

    def _get_optimizer(self):
        def _get_opt(name, init_lr):
            lr = symbf.get_scalar_var('learning_rate/'+name, init_lr, summary=True)
            opt = tf.train.AdamOptimizer(lr)
            logger.info("create opt {}".format(name))
            gradprocs = [
                # MapGradient(lambda grad: tf.Print(grad, [grad], 'grad {}='.format(grad.op.name), summarize=4)),
                MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1), regex='^actor/.*'),
                MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.05), regex='^critic/.*'),
                # GlobalNormClip(40.),
                SummaryGradient(),
            ]
            opt = optimizer.apply_grad_processors(opt, gradprocs)
            return opt
        return _get_opt('actor', INIT_LEARNING_RATE_A), _get_opt('critic', INIT_LEARNING_RATE_C)

    def get_cost(self):
        cost = self._get_cost()
        return cost

from tensorpack.callbacks.base import Callback
from tensorpack.predict.concurrency import MultiThreadAsyncPredictor
from tensorpack.utils.serialize import dumps


class MySimulatorMaster(SimulatorMaster, Callback):
    def __init__(self, pipe_c2s, pipe_s2c, model):
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
        self.M = model
        self.queue = queue.Queue(maxsize=BATCH_SIZE * 8 * 2)
        from tensorpack.utils.utils import get_rng
        self._rng = get_rng(self)

    def _setup_graph(self):
        self.async_predictor = MultiThreadAsyncPredictor(
            self.trainer.get_predictors(['state'],
                                        ['policy', 'value'],
                                        # ['Pred/policy', 'Pred/value'],
                                        PREDICTOR_THREAD), batch_size=PREDICT_BATCH_SIZE)

    def _before_train(self):
        self.async_predictor.start()

    def _on_state(self, state, ident):
        client = self.clients[ident]
        if not hasattr(client, '_cidx'):
            # client._explore = self._rng.rand()
            cidx = int(ident.decode('utf-8').replace(u'simulator-', ''))
            client._cidx = cidx
        #     if cidx % 4 == 0: client._explore = 0.

        def cb(outputs):
            try:
                policy, value = outputs.result()
            except CancelledError:
                logger.info("Client {} cancelled.".format(ident))
                return
            assert np.all(np.isfinite(policy)), policy
            action = policy
            # action = np.clip(action, -1., 1.)
            # 能否在初期得到比较好的reward决定了收敛的快慢，所以此处加入一些先验
            # 新手上路，方向盘保守一点，带点油门，不踩刹车
            # if client._cidx < SIMULATOR_PROC:
            #     if self.epoch_num <= 1:
            #         if self.local_step % 10 == 0:
            #             action[1] = self._rng.rand() * 0.5 + 0.5
            #     if action[1] < 0: action[1] = 0.
            #     if self.epoch_num <= 2:
            #         action[1] = np.clip(action[1], 0, 1.)
            #         if self.local_step % 3 == 0:
            #             action[0] *= self._rng.choice([-1., 1.])
            #             # action[0] *= (self._rng.rand() * 0.2 + 0.2) * self._rng.choice([-1., 1.])
            #         else:
            #             action[0] = np.clip(action[0], -0.2, 0.2)
            # if self._rng.rand() < client._explore:
            #     action[0] = self._rng.rand() - 0.5

            client.memory.append(TransitionExperience(
                state, action=None, reward=None, value=value))
            self.send_queue.put([ident, dumps((action,value))])
        self.async_predictor.put_task([state], cb)

    def _on_episode_over(self, ident):
        self._parse_memory(0, ident, True)

    def _on_datapoint(self, ident):
        client = self.clients[ident]
        if len(client.memory) == LOCAL_TIME_MAX + 1:
            R = client.memory[-1].value
            self._parse_memory(R, ident, False)

    def _parse_memory(self, init_r, ident, isOver):
        client = self.clients[ident]
        mem = client.memory
        if not isOver:
            last = mem[-1]
            mem = mem[:-1]

        def discount(x, gamma):
            from scipy.signal import lfilter
            return lfilter(
                [1], [1, -gamma], x[::-1], axis=0)[::-1]
        rewards_plus = np.asarray([m.reward for m in mem] + [float(init_r)])
        discounted_rewards = discount(rewards_plus, GAMMA)[:-1]
        values_plus = np.asarray([m.value for m in mem] + [float(init_r)])
        rewards = np.asarray([m.reward for m in mem])
        advantages = rewards + GAMMA * values_plus[1:] - values_plus[:-1]

        for idx, k in enumerate(mem):
            self.queue.put([k.state, k.action, discounted_rewards[idx], advantages[idx]])
        # mem.reverse()
        # R = float(init_r)
        # for idx, k in enumerate(mem):
        #     R = k.reward + GAMMA * R
        #     # R = np.clip(k.reward, -1, 1) + GAMMA * R
        #     self.queue.put([k.state, k.action, R])

        if not isOver:
            client.memory = [last]
        else:
            client.memory = []


from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow.raw import DataFromQueue
from tensorpack.train.config import TrainConfig
from tensorpack.callbacks.saver import ModelSaver
from tensorpack.callbacks.graph import RunOp
from tensorpack.callbacks.param import ScheduledHyperParamSetter, HumanHyperParamSetter, HyperParamSetterWithFunc
from tensorpack.callbacks.concurrency import StartProcOrThread
from tensorpack.tfutils import sesscreate
from tensorpack.tfutils.common import get_default_sess_config
def get_config():
    M = Model()

    name_base = str(uuid.uuid1())[:6]
    PIPE_DIR = os.environ.get('TENSORPACK_PIPEDIR', '/tmp/.ipcpipe').rstrip('/')
    if not os.path.exists(PIPE_DIR): os.makedirs(PIPE_DIR)
    else: os.system('rm -f {}/sim-*'.format(PIPE_DIR))
    namec2s = 'ipc://{}/sim-c2s-{}'.format(PIPE_DIR, name_base)
    names2c = 'ipc://{}/sim-s2c-{}'.format(PIPE_DIR, name_base)
    # AgentTorcs * SIMULATOR_PROC, AgentReplay * SIMULATOR_PROC
    procs = [MySimulatorWorker(k, namec2s, names2c) for k in range(SIMULATOR_PROC*2)]
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    master = MySimulatorMaster(namec2s, names2c, M)
    dataflow = BatchData(DataFromQueue(master.queue), BATCH_SIZE)

    class CBSyncWeight(Callback):



        def _after_run(self,ctx,_):
            if self.local_step > 1 and self.local_step % SIMULATOR_PROC ==0:
                # print("before step ",self.local_step)
                return [M._td_sync_op]

        def _before_run(self, ctx):

            if self.local_step % 10 == 0:
                return [M._sync_op,M._td_sync_op]
            if self.local_step % SIMULATOR_PROC ==0 and 0:
                return [M._td_sync_op]

    import functools
    return TrainConfig(
        model=M,
        dataflow=dataflow,
        callbacks=[
            ModelSaver(),
            HyperParamSetterWithFunc(
                'learning_rate/actor',
                functools.partial(M._calc_learning_rate, 'actor')),
            HyperParamSetterWithFunc(
                'learning_rate/critic',
                functools.partial(M._calc_learning_rate, 'critic')),

            # ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            # HumanHyperParamSetter('learning_rate'),
            # HumanHyperParamSetter('entropy_beta'),
            # ScheduledHyperParamSetter('actor/sigma_beta_accel', [(1, 0.2), (2, 0.01), (3, 1e-3), (4, 1e-4)]),
            # ScheduledHyperParamSetter('actor/sigma_beta_steering', [(1, 0.1), (2, 0.01), (3, 1e-3), (4, 1e-4)]),
            master,
            StartProcOrThread(master),
            CBSyncWeight(),
            # CBTDSyncWeight()
            # PeriodicTrigger(Evaluator(
            #     EVAL_EPISODE, ['state'], ['policy'], get_player),
            #     every_k_epochs=3),
        ],
        session_creator=sesscreate.NewSessionCreator(
            config=get_default_sess_config(0.5)),
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
    )


if __name__ == '__main__':
    import docopt, os
    args = docopt.docopt(
'''
Usage:
    main.py train [--gpu GPU] [options]
    main.py infer  [options]
    main.py test  [options]

Options:
    -h --help                   Show the help
    --version                   Show the version
    --gpu GPU                   comma separated list of GPU(s)
    --load MODELWEIGHTS         load weights from file
    --simulators SIM            simulator count             [default: 16]
    --debug_mode                set debug mode
    --a3c_instance_idx IDX      set a3c_instance_idx            [default: 0]
    --continue                  continue mode, load saved weights
    --tfdbg
    --log LEVEL                 log level                       [default: info]
    --target TARGET             test target
    --fake_agent                use fake agent to debug          
''', version='0.1')

    if args['train']:
        if args['--gpu']:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(sorted(list(set(args['--gpu'].split(',')))))
        if args['--fake_agent']:
            clsAgent = AgentFake
        # os.system('killall -9 torcs-bin')
        summary_writer=tf.summary.FileWriter('/tmp/torcs_log')

        dirname = '/tmp/torcs/trainlog'
        from tensorpack.utils import logger
        logger.set_logger_dir(dirname, action='k' if args['--continue'] else 'b')
        logger.info("Backup source to {}/source/".format(logger.LOG_DIR))
        source_dir = os.path.dirname(__file__)
        os.system('rm -f {}/sim-*; mkdir -p {}/source; rsync -a --exclude="core*" --exclude="cmake*" --exclude="build" {} {}/source/'
                  .format(source_dir, logger.LOG_DIR, source_dir, logger.LOG_DIR))
        if not args['--continue']:
            os.system('rm -rf /tmp/torcs/memory')
        os.system('rm -f /tmp/torcs_run/*.pid')

        logger.info("Create simulators, please wait...")
        clsAgent.startNInstance(SIMULATOR_PROC)

        from tensorpack.utils.gpu import get_nr_gpu
        from tensorpack.train.feedfree import QueueInputTrainer
        from tensorpack.train.multigpu import AsyncMultiGPUTrainer
        from tensorpack.tfutils.sessinit import get_model_loader
        nr_gpu = get_nr_gpu()
        # trainer = QueueInputTrainer
        assert(nr_gpu > 0)
        if nr_gpu > 1:
            predict_tower = list(range(nr_gpu))[-nr_gpu // 2:]
        else:
            predict_tower = [0]
        PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
        train_tower = list(range(nr_gpu))[:-nr_gpu // 2] or [0]
        logger.info("[Batch-A3C] Train on gpu {} and infer on gpu {}".format(
            ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))
        # if len(train_tower) > 1:
        #     trainer = AsyncMultiGPUTrainer
        from autodrive.trainer.base import MyMultiGPUTrainer
        trainer = MyMultiGPUTrainer
        config = get_config()
        if os.path.exists(logger.LOG_DIR + '/checkpoint'):
            from tensorpack.tfutils.sessinit import SaverRestore
            config.session_init = SaverRestore(logger.LOG_DIR + '/checkpoint')
        elif args['--load']:
            config.session_init = get_model_loader(args['--load'])
        config.tower = train_tower
        config.predict_tower = predict_tower

        trainer(config).train()
    elif args['infer']:
        assert args['--load'] is not None
        from tensorpack.predict.config import PredictConfig
        from tensorpack.tfutils.sessinit import get_model_loader

        cfg = PredictConfig(
            model=Model(),
            session_init=get_model_loader(args['--load']),
            input_names=['state'],
            output_names=['policy'])
        if args['--target'] == 'play':
            pass
            # play_model(cfg, get_player(viz=0.01))
        # elif args.task == 'eval':
        #     eval_model_multithread(cfg, args.episode, get_player)
        # elif args.task == 'gen_submit':
        #     play_n_episodes(
        #         get_player(train=False, dumpdir=args.output),
        #         OfflinePredictor(cfg), args.episode)
            # gym.upload(output, api_key='xxx')
    elif args['test']:
        if args['--target'] == 'agent':
            from autodrive.agent.torcs import AgentTorcs
        pass
