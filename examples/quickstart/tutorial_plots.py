"""Tutorial-only diagrams and plots; no model loading or inference."""

import numpy as np
import matplotlib.pyplot as plt



def plot_query_pairs(pairs):
    """Draw each query independently; arrows do not imply biological evidence."""
    shown = pairs.head(10).reset_index(drop=True)
    if shown.empty:
        raise ValueError('At least one query is needed for the diagram.')
    count = len(shown)
    fig, ax = plt.subplots(figsize=(8, max(3, .55 * count + 1.5)))
    ax.set(xlim=(0, 1), ylim=(-.7, count))
    ax.axis('off')
    ax.text(.25, count - .2, 'Source (gene_i)', ha='center', fontsize=11, color='#176c66')
    ax.text(.75, count - .2, 'Target (gene_j)', ha='center', fontsize=11, color='#315a8b')
    for row, (source, target) in enumerate(shown[['gene_i', 'gene_j']].to_numpy()):
        y = count - 1 - row
        ax.text(.04, y, str(row + 1), ha='center', va='center', color='#687789')
        for x, label, color in ((.25, source, '#dceeea'), (.75, target, '#e7eff9')):
            ax.text(x, y, label, ha='center', va='center', fontsize=11,
                    bbox={'boxstyle': 'round,pad=0.4', 'fc': color, 'ec': 'none'})
        ax.annotate('', xy=(.62, y), xytext=(.38, y),
                    arrowprops={'arrowstyle': '->', 'lw': 1.7, 'color': '#526477'})
    fig.suptitle('Input queries — arrows are questions, not validated interactions', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    return fig


def plot_scores(predictions, top_k=10, title='Model scores'):
    """Plot actual computed logits and calibrated scores with shared ordering."""
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError('top_k must be a positive integer.')
    if predictions.empty:
        raise ValueError('At least one prediction is needed for plotting.')
    shown = predictions.sort_values('calibrated_probability', ascending=False).head(top_k)
    labels = [f'{a} → {b}' for a, b in shown[['gene_i', 'gene_j']].to_numpy()]
    y = np.arange(len(shown))
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.2, .45 * len(shown) + 1.5)), sharey=True)
    for ax, column, color in zip(axes, ['logit', 'calibrated_probability'], ['#277d78', '#4b78a8']):
        values = shown[column].to_numpy()
        ax.barh(y, values, height=.6, color=color)
        ax.set_axisbelow(True)
        ax.grid(axis='x', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        for row, value in enumerate(values):
            # Labels use axis-relative x to avoid clipping scores near 1.
            label = f'{value:.4g}' if column == 'calibrated_probability' else f'{value:.4f}'
            ax.text(.98, row, label, transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=9,
                    bbox={'fc': 'white', 'ec': 'none', 'alpha': .9, 'pad': 1})
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color='#526477', lw=.8)
    axes[0].set_xlabel('Raw decoder score (logit)')
    axes[0].margins(x=.15)
    axes[1].set(xlim=(0, 1), xlabel='Calibrated model score (0–1)')
    fig.suptitle(title, fontsize=13)
    fig.text(.5, .02, 'Fixed embryo1 context • illustrative queries, not independent biological validation',
             ha='center', fontsize=9, color='#526477')
    fig.tight_layout(rect=(0, .06, 1, .94))
    return fig
