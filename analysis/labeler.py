import zarr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

def interactive_label_clusters_ui(
    df: pd.DataFrame,
    output_folder: Path,
    label_options=None,
    config: dict = None
):
    """
    df must have columns ['dataset_id','window_id','GMM_cluster','start_time','end_time',...]
    zarr_folder contains segments.zarr with groups by dataset_id
    """
    # open the store
    store = zarr.storage.LocalStore(str(config['project_name'] + "/segments.zarr"))
    root  = zarr.open(store, mode="r")

    sampling_rate = config.get('sampling_rate', 500)  # default to 1000 Hz if not specified
    clusters = sorted(df['GMM_cluster'].unique())
    n = len(clusters)
    figs   = [None]*n
    labels = [None]*n
    idx = 0

    if label_options is None:
        label_options = ["seizure","normal","artifact","unknown"]

    # widgets
    out = widgets.Output()
    status = widgets.Label()
    dd = widgets.Dropdown(options=clusters, description="Cluster")
    btn_next = widgets.Button(description="Next ▶", button_style="success")
    btn_back = widgets.Button(description="◀ Back", button_style="warning")
    nav = widgets.HBox([btn_back, btn_next, dd])
    lbl_box = widgets.VBox()

    def render_labels(sel=None):
        btns=[]
        for i,opt in enumerate(label_options):
            prefix = "✅" if i==sel else "⬜"
            b = widgets.Button(description=f"{prefix} {opt}", layout=widgets.Layout(width="120px"))
            b.on_click(lambda b,i=i: select_label(i))
            btns.append(b)
        lbl_box.children = btns

    def select_label(i):
        state['sel'] = i
        render_labels(i)

    def load_and_show():
        nonlocal idx
        cl = clusters[idx]
        status.value = f"Cluster {idx+1}/{n}  ({cl})"
        dd.value = cl
        sub = df[df['GMM_cluster']==cl].reset_index(drop=True)
        k = min(len(sub), 20)
        sample = sub.sample(k, random_state=0).reset_index(drop=True)

        fig, axes = plt.subplots(5,4,figsize=(18,14))
        axes = axes.flatten()
        for ax in axes[k:]:
            ax.axis("off")

        for i,row in sample.iterrows():
            ds = row['dataset_id']
            w  = int(row['window_id'])
            seg = root[ds][w]  # shape (window_length,)
            # we assume seg already has context, so just plot seg
            t = np.arange(len(seg)) / sampling_rate
            ax = axes[i]
            ax.plot(t, seg, color="black", lw=0.8)
            # highlight the center region
            ax.axvspan(0, (row['end_time']-row['start_time'])/1000, color="red", alpha=0.3)
            ax.axis("off")

        plt.suptitle(f"Cluster {cl} Examples", fontsize=18)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        figs[idx] = fig

        # restore previous selection
        prev = labels[idx][1] if labels[idx] else None
        sel = label_options.index(prev) if prev in label_options else None
        state['sel'] = sel
        render_labels(sel)

        with out:
            clear_output(wait=True)
            display(status, nav, lbl_box, fig)
        plt.close(fig)

    def on_next(_):
        if state['sel'] is None:
            status.value = "⚠️ Please select a label!"
            return
        labels[idx] = (clusters[idx], label_options[state['sel']])
        if idx < n-1:
            idx += 1
            state['sel'] = None
            load_and_show()
        else:
            # save PDF + CSV
            output_folder.mkdir(parents=True, exist_ok=True)
            pdf = PdfPages(output_folder/f"clusters_examples.pdf")
            for f in figs:
                pdf.savefig(f)
            pdf.close()
            pd.DataFrame(labels, columns=["GMM_cluster","label"])\
              .to_csv(output_folder/"cluster_labels.csv", index=False)
            with out:
                clear_output()
                print("✅ Done! PDF and labels saved.")
    
    def on_back(_):
        nonlocal idx
        if idx>0:
            idx -= 1
            state['sel'] = None
            load_and_show()

    def on_select(change):
        nonlocal idx
        new = change['new']
        if new!=clusters[idx]:
            idx = clusters.index(new)
            state['sel'] = None
            load_and_show()

    # link events
    state = {'sel': None}
    btn_next.on_click(on_next)
    btn_back.on_click(on_back)
    dd.observe(on_select, names='value')

    # kickoff
    display(out)
    load_and_show()
