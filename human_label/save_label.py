import os
import pickle

env_name = "halfcheetah-medium-expert-v2"
labels = []
# manually code your label to the variable `labels`.
# labels[i] = 0 if left one does better
# labels[i] = 1 if right one does better
# labels[i] = -1 if neither one is significantly better than the other

with open(os.path.join(env_name, "label_human"), "wb") as fp:
    pickle.dump(labels, fp)
