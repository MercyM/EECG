from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .actor_critic_pac_dcg_learner import PACDCGLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner


REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["pac_learner"] = PACActorCriticLearner
REGISTRY["pac_dcg_learner"] = PACDCGLearner


from .fast_q_learner import fast_QLearner
from .EA_fast_q_learner import EA_fast_QLearner
from .qplex_curiosity_vdn_learner import QPLEX_curiosity_vdn_Learner
from .EA_qplex_curiosity_vdn_learner import EA_QPLEX_curiosity_vdn_Learner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .qatten_learner import QattenLearner
from .qtran_transformation_learner import QLearner as QTranTRANSFORMATIONLearner

REGISTRY["fast_QLearner"] = fast_QLearner
REGISTRY["EA_fast_QLearner"] = EA_fast_QLearner
REGISTRY['qplex_curiosity_vdn_learner'] = QPLEX_curiosity_vdn_Learner
REGISTRY['EA_qplex_curiosity_vdn_learner'] = EA_QPLEX_curiosity_vdn_Learner
REGISTRY["qatten_learner"] = QattenLearner
REGISTRY["qtran_transformation_learner"] = QTranTRANSFORMATIONLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner