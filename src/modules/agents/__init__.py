from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent


REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent

from .rnn_fast_agent import RNNFastAgent
REGISTRY["rnn_fast"] = RNNFastAgent


from .sco_agent import SCOAgent
from .simple_agent import SimpleAgent
from .rnd_nn_agent import RND_nn_Agent
from .rnd_history_agent import RNDHistoryAgent
from .rnn_individualQ_agent import RNN_individualQ_Agent
from .rnd_fast_history_agent import RND_Fast_historyAgent


REGISTRY["sco"] = SCOAgent
REGISTRY["simple"] = SimpleAgent
REGISTRY["rnd_nn"] = RND_nn_Agent
REGISTRY["rnd_history"] = RNDHistoryAgent
REGISTRY["rnn_individualQ"] = RNN_individualQ_Agent
REGISTRY["rnd_fast_history"] = RND_Fast_historyAgent