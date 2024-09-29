REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC

from .fast_controller import FastMAC

REGISTRY["fast_mac"] = FastMAC

from .mmdp_controller import MMDPMAC
from .qsco_controller import qsco_MAC
from .rnd_state_predictor import RND_state_predictor
from .rnd_predictor import RNDpredictor
from .fast_rnd_predictor import RNDfastpredictor

REGISTRY["mmdp_mac"] = MMDPMAC
REGISTRY["qsco_mac"] = qsco_MAC
REGISTRY["nn_predict"] = RND_state_predictor

REGISTRY["predict"] = RNDpredictor
REGISTRY["fast_predict"] = RNDfastpredictor