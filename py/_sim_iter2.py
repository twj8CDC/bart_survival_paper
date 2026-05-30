import matplotlib.pyplot as plt
import importlib
import _functions1 as fn
import _conditions1 as cn
import _plot_fx as pltf
plt.ioff()
importlib.reload(pltf)
importlib.reload(fn)
importlib.reload(cn)

# complex regressions
def iter_simulation_complex12(iters, n, seed_addl, scenario, SPLIT_RULES, model_dict, sampler_dict, plot_all):
    """_summary_

    Args:
        iters (_type_): _description_
        n (_type_): _description_
        scenario (_type_): _description_
        SPLIT_RULES (_type_): _description_
        model_dict (_type_): _description_
        sampler_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    meta_lst = []
    sv_true_lst0 = []
    cph_sv_lst0 = []
    pb_sv_lst0 = []
    r_sv_lst0 = []
    fig_l = []
        
    strt = n*seed_addl
    end = strt + iters
    with open("../output.txt", "w") as f:
        print("Starting complex 1")
    for i in range(strt, end):
        with open("../output.txt", "a") as f:
            print(f"ITERATION************** {i}", file=f)
        print(f"ITERATION************** {i}")
        meta, sv_true, cph_sv, pb_sv, r_sv = fn.sim_3s(seed=i, n=n, scenario_=scenario, SPLIT_RULES=SPLIT_RULES, model_dict=model_dict, sampler_dict=sampler_dict)

        uniq_t = meta["qnt_t"][0]
        uniq_idx = meta["qnt_t"][1]

        true_t = meta["true_t"]
        assert((true_t[uniq_idx] == uniq_t).all)

        sv_t_0 = sv_true["sv_true"][:,uniq_idx] # this is obs
        cph_sv_0 = cph_sv # this is obs
        p_sv_0 = pb_sv # this is already the mean, this is obs
        r_sv_0 = r_sv # this is already the mean, this is obs
        # there is a unsafe error that can occur if the sim event times don't map 1:1 to the qnt_times
        # this occurs if like the 95% t is 2 and the minimum event time is 4.
        assert(sv_t_0.shape == cph_sv_0.shape == p_sv_0.shape == r_sv_0.shape)    

        meta_lst.append(meta)
        sv_true_lst0.append(sv_t_0)
        cph_sv_lst0.append(cph_sv_0)
        pb_sv_lst0.append(p_sv_0)
        r_sv_lst0.append(r_sv_0)

        if plot_all:
            fig = pltf.plots3(uniq_t, sv_t_0, cph_sv_0, p_sv_0, r_sv_0)
            title = f"{scenario['type']}, n {n}"
            fig.suptitle(title)
            fig_l.append(fig)
            if i < strt+10:
                ttl = scenario["type"]
                namesss = f"../figs/{ttl}_{i}.png"
                fig.savefig(namesss)
    
    if not plot_all:
        fig = pltf.plots3(meta, sv_t_0, cph_sv_0, p_sv_0, r_sv_0)
        title = f"{scenario['type']}, n {n}"
        fig.suptitle(title)
        fig_l.append(fig)

    c,p,r = fn.get_metrics3(
        sv_true_lst0,
        cph_sv_lst0,
        pb_sv_lst0,
        r_sv_lst0,
    )
    cens1 = [m["cens_perc"] for m in meta_lst]
    cens2 = [m["cens_perc2"] for m in meta_lst]
    # cens1 = np.array([m["cens_perc"] for m in meta_lst]).mean()
    # cens2 = np.array([m["cens_perc2"] for m in meta_lst]).mean()
    
    return meta_lst, (strt, end), (cens1,cens2), c,p,r, fig_l


