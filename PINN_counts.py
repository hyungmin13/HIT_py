#%%
import jax
import jax.numpy as jnp
from jax import random
from time import time
import optax
from jax import value_and_grad
from functools import partial
from jax import jit
from tqdm import tqdm
from typing import Any
from flax import struct
from flax.serialization import to_state_dict, from_state_dict
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as st
import h5py
from tqdm import tqdm
class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

class PINNbase:
    def __init__(self,c):
        c.get_outdirs()
        c.save_constants_file()
        self.c=c
class PINN(PINNbase):
    def test(self):
        all_params = {"domain":{}, "data":{}, "network":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        all_params["network"] = self.c.network.init_params(**self.c.network_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)
        optimiser = self.c.optimization_init_kwargs["optimiser"](self.c.optimization_init_kwargs["learning_rate"])
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        all_params = self.c.problem.constraints(all_params)
        valid_data = self.c.problem.exact_solution(all_params)
        model_fn = c.network.network_fn
        return all_params, model_fn, train_data, valid_data

#%%
if __name__ == "__main__":
    from PINN_domain import *
    from PINN_trackdata import *
    from PINN_network import *
    from PINN_constants import *
    from PINN_problem import *
    checkpoint_fol = "run02"
    path = "results/summaries/"
    with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','rb') as f:
        a = pickle.load(f)

    a['data_init_kwargs']['path'] = '/home/hgf_dlr/hgf_dzj2734/HIT/Particles/'
    a['problem_init_kwargs']['path_s'] = '/home/hgf_dlr/hgf_dzj2734/HIT/IsoturbFlow.mat'
    a['problem_init_kwargs']['problem_name'] = 'HIT'
    with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','wb') as f:
        pickle.dump(a,f)
    f.close()
    values = list(a.values())

    c = Constants(run = values[0],
                domain_init_kwargs = values[1],
                data_init_kwargs = values[2],
                network_init_kwargs = values[3],
                problem_init_kwargs = values[4],
                optimization_init_kwargs = values[5],)
    run = PINN(c)
    with open(run.c.model_out_dir + "saved_dic_640000.pkl","rb") as f:
        a = pickle.load(f)
    all_params, model_fn, train_data, valid_data = run.test()

    model = Model(all_params["network"]["layers"], model_fn)
    all_params["network"]["layers"] = from_state_dict(model, a).params

    output_shape = (129,129,129)
    total_spatial_error = []
    t_pos_un = np.concatenate([train_data['pos'][:,i:i+1]*all_params["domain"]["in_max"][0,i]
                                 for i in range(4)],1).reshape(-1,4)

    t_pos_c = t_pos_un - np.array([0,0.05,0.05,0.05]).reshape(-1,4)
    t_pos_c = np.sqrt(t_pos_c[:,1]**2+t_pos_c[:,2]**2+t_pos_c[:,3]**2)
    t_pos_un = t_pos_un.reshape(-1,4)

    counts, bins, bars = plt.hist(t_pos_c, bins=50)

#%% 수치해석법으로 bins에 따라 볼륨 중앙에 중심을 둔 구와 볼륨 사이의 겹치는 부피를 구하는 코드
    """
    from scipy.integrate import tplquad
    L = 0.1
    W = 0.1
    H = 0.1
    def sphere_condition(x,y,z):
        return x**2+y**2+z**2<r**2
    def intergrand(x,y,z):
        return 1 if sphere_condition(x,y,z) else 0
    volumes = []
    for i in range(len(bins)):
        print(i)
        r = bins[i]
        print(r)
        volume, error = tplquad(intergrand, -L/2, L/2, lambda x:-W/2, lambda x:W/2, lambda x,y:-H/2, lambda x,y:H/2)
        volumes.append(volume)
    sub_volumes = np.array(volumes[1:])-np.array(volumes[:-1])
    c_vol_avg = np.array(counts)/np.array(sub_volumes)
    #plt.hist(bins[:-1], bins,weights=c_vol_avg)
    #plt.show()
    with open("datas/sub_volumes.pkl","wb") as f:
        pickle.dump(sub_volumes,f)
    f.close()
    with open("datas/counts.pkl","wb") as f:
        pickle.dump(counts,f)
    f.close()
    """

# %% 격자해석법으로 bins에 따라 볼륨 중앙에 중심을 둔 구와 볼륨 사이의 겹치는 부피를 구하는 코드
    import numpy as np
    L = 0.1
    W = 0.1
    H = 0.1
    grid_resolution = 0.0001
    grid_volume = grid_resolution**3
    x = np.arange(-L/2, L/2, grid_resolution)
    y = np.arange(-W/2, W/2, grid_resolution)
    z = np.arange(-H/2, H/2, grid_resolution)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    volumes = []
    for i in range(len(bins)):
        print(i)
        r = bins[i]
        inside_sphere = xx**2+yy**2+zz**2<r**2
        volumes.append(np.sum(inside_sphere)*grid_volume)

    subvolumes = np.array(volumes[1:])-np.array(volumes[:-1])
    c_vol_avg = np.array(counts)/subvolumes
    p_distribution = {"counts":counts, "subvolumes":subvolumes, "vol_avg":c_vol_avg}
    if os.path.isdir("datas/"+checkpoint_fol):
        pass
    else:
        os.mkdir("datas/"+checkpoint_fol)
    with open("datas"+"/counts.pkl","wb") as f:
        pickle.dump(p_distribution,f)
    f.close()
# %%
