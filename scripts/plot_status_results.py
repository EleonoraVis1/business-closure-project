"""Plot saved recorded-status experiments without retraining or rescoring reviews."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/status_experiments'
FIGURES = OUT / 'figures'
COLORS = {'logistic_regression': '#176b87', 'random_forest': '#d27c2c'}
LABELS = {'logistic_regression': 'Logistic Regression', 'random_forest': 'Random Forest'}
GROUPS = ['A_basic', 'B_volume', 'C_temporal', 'D_activity', 'E_activity_ratings', 'F_activity_sentiment', 'G_full']
GROUP_LABELS = ['A: Basic', 'B: + annual volume', 'C: Temporal volume', 'D: Activity', 'E: Activity + ratings', 'F: Activity + sentiment', 'G: Full']

def save(fig, name):
    fig.savefig(FIGURES / f'{name}.png', dpi=190, bbox_inches='tight')
    plt.close(fig)

def ablations():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    for row, population in enumerate(['main', 'active']):
        frame = pd.read_csv(OUT / population / 'model_comparison.csv')
        groups = GROUPS if population == 'main' else GROUPS[3:]
        labels = GROUP_LABELS if population == 'main' else GROUP_LABELS[3:]
        for col, metric in enumerate(['roc_auc_closed', 'average_precision_closed']):
            ax = axes[row, col]
            for model, color in COLORS.items():
                values = frame.loc[frame.model.eq(model)].set_index('feature_group').loc[groups, metric]
                ax.plot(range(len(groups)), values, 'o-', color=color, label=LABELS[model], lw=2)
            baseline = frame.loc[frame.feature_group.eq('majority'), metric].iloc[0]
            ax.axhline(baseline, color='#777777', linestyle=':', label='Majority baseline')
            ax.set_xticks(range(len(groups)), labels, rotation=30, ha='right', fontsize=9)
            ax.set_title(f"{'All restaurants' if population == 'main' else '2021-active restaurants'} · {'ROC-AUC' if col == 0 else 'Closed-class average precision'}")
            ax.set_ylim(0, 1)
            ax.set_ylabel('Held-out score')
            ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(loc='lower right', fontsize=9)
    fig.suptitle('Feature-group comparisons for recorded Yelp status', fontsize=15)
    save(fig, 'feature_group_comparison')

def increments():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    names = {
        ('A_basic','B_volume'): 'Annual volume beyond basic',
        ('B_volume','C_temporal'): 'Temporal vs annual volume',
        ('C_temporal','D_activity'): 'Add check-ins and tips',
        ('D_activity','E_activity_ratings'): 'Add ratings to activity',
        ('D_activity','F_activity_sentiment'): 'Add sentiment to activity',
        ('E_activity_ratings','G_full'): 'Add sentiment after ratings'}
    for ax, population in zip(axes, ['main', 'active']):
        data = pd.read_csv(OUT / population / 'incremental_contributions.csv')
        data = data.loc[data.metric.eq('roc_auc_closed')]
        pairs = list(names) if population == 'main' else list(names)[3:]
        for model, offset in [('logistic_regression', -.12), ('random_forest', .12)]:
            group = data.loc[data.model.eq(model)].set_index(['before', 'after'])
            rows = group.loc[pairs]
            delta, low, high = (rows[c].to_numpy() for c in ['delta', 'ci_low', 'ci_high'])
            # Draw intervals directly: percentile intervals need not contain point estimates.
            y = np.arange(len(pairs)) + offset
            ax.hlines(y, low, high, color=COLORS[model], lw=2)
            ax.plot(delta, y, 'o', color=COLORS[model], label=LABELS[model])
        ax.axvline(0, color='#555555', linestyle='--', lw=1)
        ax.set_yticks(range(len(pairs)), [names[p] for p in pairs], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Change in held-out ROC-AUC')
        ax.set_title('All restaurants' if population == 'main' else '2021-active restaurants')
        ax.grid(axis='x', alpha=.2)
        if population == 'main':
            # The annual-volume increment is large; preserve it and add exact near-zero values in CSV.
            ax.set_xlim(-.025, .355)
        if population == 'main':
            ax.legend(fontsize=8, loc='best')
    fig.suptitle('Incremental feature value · paired 95% bootstrap intervals', fontsize=15)
    fig.supxlabel('500 stratified holdout resamples; conditional on fitted models; exploratory intervals', fontsize=10)
    save(fig, 'incremental_auc')
    # A separate close-up makes the small sentiment increments readable without hiding the large volume effect.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
    pairs = [('D_activity','F_activity_sentiment'), ('E_activity_ratings','G_full')]
    for ax, population in zip(axes, ['main', 'active']):
        data = pd.read_csv(OUT / population / 'incremental_contributions.csv')
        data = data.loc[data.metric.eq('roc_auc_closed')]
        for model, offset in [('logistic_regression', -.1), ('random_forest', .1)]:
            rows = data.loc[data.model.eq(model)].set_index(['before','after']).loc[pairs]
            y = np.arange(2)+offset
            ax.hlines(y, rows.ci_low, rows.ci_high, color=COLORS[model], lw=2)
            ax.plot(rows.delta, y, 'o', color=COLORS[model], label=LABELS[model])
        ax.axvline(0, color='#555555', linestyle='--', lw=1)
        ax.set_yticks([0,1], ['Sentiment beyond activity','Sentiment beyond activity + ratings'], fontsize=9)
        ax.invert_yaxis(); ax.set_xlabel('Change in held-out ROC-AUC')
        ax.set_title('All restaurants' if population == 'main' else '2021-active restaurants')
        ax.grid(axis='x', alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    axes[0].legend(fontsize=8)
    fig.suptitle('Sentiment contribution · paired 95% bootstrap intervals', fontsize=15)
    save(fig, 'sentiment_contribution')

def curves():
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout='constrained')
    for row, population in enumerate(['main','active']):
        data = pd.read_csv(OUT / population / 'predictions.csv')
        metrics = pd.read_csv(OUT / population / 'model_comparison.csv')
        for model, color in COLORS.items():
            p = data.loc[data.model.eq(model) & data.feature_group.eq('G_full')]
            m = metrics.loc[metrics.model.eq(model) & metrics.feature_group.eq('G_full')].iloc[0]
            fpr, tpr, _ = roc_curve(p.actual_closed, p.probability_closed)
            precision, recall, _ = precision_recall_curve(p.actual_closed, p.probability_closed)
            axes[row,0].plot(fpr,tpr,color=color,label=f"{LABELS[model]} (AUC {m.roc_auc_closed:.3f})",lw=2)
            axes[row,1].plot(recall,precision,color=color,label=f"{LABELS[model]} (AP {m.average_precision_closed:.3f})",lw=2)
        axes[row,0].plot([0,1],[0,1],ls=':',color='#777777',label='Chance')
        axes[row,1].axhline(p.actual_closed.mean(),ls=':',color='#777777',label=f'Closed prevalence {p.actual_closed.mean():.1%}')
        axes[row,0].set(xlabel='False positive rate (open called closed)',ylabel='Closed-class recall')
        axes[row,1].set(xlabel='Closed-class recall',ylabel='Closed-class precision')
        for ax, title in zip(axes[row], ['ROC curve','Precision–recall curve']):
            ax.set_title(f"{'All restaurants' if population == 'main' else '2021-active restaurants'} · {title}")
            ax.set_xlim(0,1); ax.set_ylim(0,1.02); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('Full models · held-out recorded-status classification', fontsize=15)
    save(fig, 'classification_curves')

def confusion():
    fig, axes = plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for row,population in enumerate(['main','active']):
        data = pd.read_csv(OUT / population / 'model_comparison.csv')
        for col,model in enumerate(COLORS):
            m=data.loc[data.model.eq(model)&data.feature_group.eq('G_full')].iloc[0]
            counts=np.array([[m.tn_open,m.fp_closed],[m.fn_closed,m.tp_closed]], dtype=int)
            fractions=counts/counts.sum(axis=1,keepdims=True)
            ax=axes[row,col]
            ax.imshow(fractions,vmin=0,vmax=1,cmap='Blues')
            for (i,j),value in np.ndenumerate(counts):
                ax.text(j,i,f'{value:,}\n({fractions[i,j]:.1%} of actual class)',ha='center',va='center',color='white' if fractions[i,j]>.55 else '#111111',fontsize=9)
            ax.set_xticks([0,1],['Open','Closed']);ax.set_yticks([0,1],['Open','Closed'])
            ax.set(xlabel='Predicted recorded status',ylabel='Actual recorded status',title=f"{'All' if population=='main' else '2021-active'} · {LABELS[model]}")
    fig.suptitle('Full-model confusion matrices · threshold 0.5',fontsize=15)
    save(fig,'confusion_matrices')

def main():
    FIGURES.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    ablations();increments();curves();confusion()
    print('Saved five figures to',FIGURES)

if __name__=='__main__':main()
