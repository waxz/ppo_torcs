#coding: utf-8

import numpy as np
import abc
from tensorpack.RL.gymenv import GymEnv
from ..utils import logger

from .memory import Memory, MemorySaver
from ..utils import logger
class AgentBase(GymEnv):
    def __init__(self, agentIdent, is_train=False, auto_restart = True, **kwargs):
        # super(AgentBase, self).__init__(name='torcs')
        self.auto_restart = auto_restart
        self._isTrain = is_train
        self._agentIdent = agentIdent
        self._kwargs = kwargs
        self._init()

    def _init(self):
        logger.info("[{}]: agent init, isTrain={}".format(self._agentIdent, self._isTrain))
        self._episodeCount = -1
        from tensorpack.utils.utils import get_rng
        self._rng = get_rng(self)
        from tensorpack.utils.stats import StatCounter
        self.reset_stat()
        self.rwd_counter = StatCounter()
        self._memorySaver = None
        save_dir = self._kwargs.pop('save_dir', None)
        if save_dir is not None:
            self._memorySaver = MemorySaver(save_dir,
                                            self._kwargs.pop('max_save_item', 3),
                                            self._kwargs.pop('min_save_score', None),
                                            )
        self.restart_episode()
        pass

    def restart_episode(self):
        self.rwd_counter.reset()
        self.__ob = self.reset()

    def finish_episode(self):
        score = self.rwd_counter.sum
        self.stats['score'].append(score)
        logger.info("episode finished, rewards = {:.3f}, episode = {}, steps = {}"
                    .format(score, self._episodeCount, self._episodeSteps))

    def current_state(self):
        return self.__ob

    def reset(self):
        self._episodeCount += 1
        ret = self._reset()
        self._episodeRewards = 0.
        self._episodeSteps = 0
        if self._memorySaver:
            self._memorySaver.createMemory(self._episodeCount)
        logger.info("restart, episode={}".format(self._episodeCount))
        return ret

    @abc.abstractmethod
    def _reset(self):
        pass

    def action(self, pred):
        ob, act, r, isOver, info = self._step(pred)
        self.rwd_counter.feed(r)
        if self._memorySaver:
            self._memorySaver.addCurrent(ob, act, r, isOver)
        self.__ob = ob
        self._episodeSteps += 1
        self._episodeRewards += r
        if isOver:
            self.finish_episode()
            if self.auto_restart:
                self.restart_episode()
        return act, r, isOver

    @abc.abstractmethod
    def _step(self, action):
        raise NotImplementedError

    def get_action_space(self):
        raise NotImplementedError
        # spc = self.gymenv.action_space
        # assert isinstance(spc, gym.spaces.discrete.Discrete)
        # return DiscreteActionSpace(spc.n)

class AgentMemoryReplay(AgentBase):
    def _init(self):
        load_dir = self._kwargs.pop('load_dir', '/tmp/agent_reply')
        import os
        if not os.path.exists(load_dir): os.makedirs(load_dir)
        self._load_dir = load_dir
        self._max_save_item = self._kwargs.pop('max_save_item', 3)
        self._memoryStartIdx = self._kwargs.pop('memory_start_idx', 0)
        self._memoryBests = []  # type: list[Memory]
        self._memoryCurrent = None  # type: Memory
        self._scanSavedMemory()
        self._flagEnd = False
        super(AgentMemoryReplay, self)._init()
        assert (self._memorySaver is None)

    def _scanSavedMemory(self):
        import os
        import filelock
        # from ..utils import logger
        lockfile = self._load_dir + '/.lock'
        try:
            locker = filelock.FileLock(lockfile)
            with locker.acquire(3):
                files = []
                for directory, dirnames, filenames in os.walk(self._load_dir):
                    files += [os.path.join(directory, n) for n in filenames if n.endswith('.pkl')]
                if len(files) > 0:
                    for filename in files:
                        episode = int(os.path.basename(filename).replace('.pkl', ''))
                        if episode in [m.episode for m in self._memoryBests]: continue
                        with open(filename, 'rb') as f:
                            import pickle
                            m = pickle.load(f)
                            assert(isinstance(m, Memory))
                            if len(self._memoryBests) >= self._max_save_item:
                                idx_low = np.argmin([m.score for m in self._memoryBests])
                                assert(self._memoryBests[idx_low].score < m.score), 'load memory score {} < best memory score {}'.format(m.score, self._memoryBests[idx_low].score)
                                logger.info("remove memory, score={:.3f}, episode={}"
                                            .format(self._memoryBests[idx_low].score, self._memoryBests[idx_low].episode))
                                del self._memoryBests[idx_low]
                            self._memoryBests.append(m)

                            logger.info("load memory, score={:.3f}, episode={}, bests={}".format(m.score, m.episode, len(self._memoryBests)))

                    return len(self._memoryBests)
        except filelock.Timeout:
            logger.error("can not lock file {}".format(lockfile))

    def _reset(self):
        super(AgentMemoryReplay, self)._reset()
        while not self._flagEnd:
            self._scanSavedMemory()
            if len(self._memoryBests) > 0:
                break
            from time import sleep
            sleep(1)
            continue
        self._memoryCurrent = m = self._memoryBests[self._rng.choice(len(self._memoryBests))]
        duration = m._timeEnd - m._timeStart
        self._timePerStep = duration.total_seconds() / m.size
        self._memoryCurIdx = 0
        if self._memoryStartIdx < 0:
            self._memoryCurIdx = self._rng.randint(0, int(m.size * 0.8))

        logger.info("select memory: episode={}, score={:.4f}, size={}, duration = {}, timePerStep={:.4f}, start={}"
                    .format(self._memoryCurrent.episode, self._memoryCurrent.score, self._memoryCurrent.size,
                            duration, self._timePerStep, self._memoryCurIdx,
                            ))
        return self._memoryCurrent[self._memoryCurIdx].state

    def _step(self, action):
        assert(self._memoryCurrent is not None)
        item = self._memoryCurrent[self._memoryCurIdx]
        memlen = self._memoryCurrent.size
        nextob = self._memoryCurrent[self._memoryCurIdx+1 if self._memoryCurIdx < (memlen -1) else -1].state
        self._memoryCurIdx += 1
        from time import sleep
        sleep(self._timePerStep)
        return nextob, item.action, item.reward, self._memoryCurIdx >= (memlen - 1), {}