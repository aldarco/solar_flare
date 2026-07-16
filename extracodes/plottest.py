# use to watch the data fitted with the ARmodel for hmm
# considers only 3 states

import matplotlib.pyplot as plt
import pandas  as pd
import sys

if len(sys.argv)==1:
	data = pd.read_csv("data_trained_diurnal_modelpreds_models_1-3-6-10-15.csv", index_col=0, parse_dates=True)
else:
	data = pd.read_csv(sys.argv[1], index_col=0, parse_dates=True)

_states_values = [0,1,2]

df_clean = data
order_values = [int(x.split("_o")[-1]) for x in df_clean.columns if x.startswith("states_") ]
order_colors = ["royalblue", "red", "darkorange"]

fig, axs = plt.subplots(nrows=len(order_values)+1, figsize=(8,10), sharex=True)
ax= axs[0]
ax.plot(df_clean.index, df_clean["NAA"].values, "k", alpha=0.5)
ax.set_ylim(-10,10)
ax2 = ax.twinx()
for s in _states_values:
	mask = df_clean["state"] == s
	markerline, stemline, baseline = ax2.stem(df_clean.index[mask], df_clean["state"].values[mask], linefmt=order_colors[s], markerfmt="s", bottom=0, label=f"state={s}")

markerline.set_markerfacecolor("none")
markerline.set_alpha(0.5)
stemline.set_alpha(0.5)
ax2.spines["right"].set_visible(False)
ax2.yaxis.set_visible(False)
ax3 = ax2.twinx()
ax3.plot(df_clean["GOES19"], color="violet", alpha=0.7)
ax3.set_yscale("log")
ax3.set_ylabel("Xray Flux")

for k, order in enumerate(order_values, start=1):
	states_max_prob = df_clean[f"states_o{order}"]
	ax = axs[k]    
	ax.plot(df_clean.index, states_max_prob, marker="s", color="firebrick", ms=4, mfc="none")
    # ax.plot(df_clean.index, X_train_norm[:, 0], "k", alpha=0.5)
	ax.grid(ls="--", lw=0.7)
plt.show()
