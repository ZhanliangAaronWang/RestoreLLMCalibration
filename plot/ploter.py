import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

def histogram_plotter_cw(TP, Total, confidence_bins):
    accuracy = {'A':[], 'B':[], 'C':[], 'D':[], 'Total':{}}
    confidences_bins = {'A':[], 'B':[], 'C':[], 'D':[], 'Total':{}}
    underconfidence_bins = {'A':0, 'B':0, 'C':0, 'D':0, 'Total':0}
    overconfidence_bins = {'A':0, 'B':0, 'C':0, 'D':0, 'Total':0}
    weight_options = {'A':[], 'B':[], 'C':[], 'D':[], 'Total':[]}
    weight_rank_options = {'A':[], 'B':[], 'C':[], 'D':[], 'Total':[]}
    

    TP2 = TP.sum(0)
    Total2 = Total.sum(0)
    confidence_bins2 = (Total * confidence_bins.nan_to_num()).sum(0)/Total2

    for option, name in enumerate(accuracy.keys()):
        if option <= 3: 
            accuracy[name] = np.array(torch.div(TP[option], Total[option]).nan_to_num())
            confidences_bins[name] = np.array(confidence_bins[option].nan_to_num())
            underconfidence = np.maximum(0, accuracy[name]-confidences_bins[name])
            overconfidence = np.maximum(0, confidences_bins[name]-accuracy[name])
            underconfidence_bins[name] = underconfidence
            overconfidence_bins[name] = overconfidence
            weight_options[name] = np.array(Total[option]/Total[option].sum())
            weight_rank_options[name] = np.argsort(weight_options[name])
        else:  # Total
            accuracy[name] = np.array(TP2/Total2)
            confidences_bins[name] = np.array(confidence_bins2.nan_to_num())
            underconfidence = np.maximum(0, accuracy[name]-confidences_bins[name])
            overconfidence = np.maximum(0, confidences_bins[name]-accuracy[name])
            underconfidence_bins[name] = underconfidence
            overconfidence_bins[name] = overconfidence
            weight_options[name] = np.array(Total2/Total2.sum())
            weight_rank_options[name] = np.argsort(weight_options[name])
    
    x_positions = np.linspace(0.0, 0.9, 10)

    data = {
        'A': {'accuracy': accuracy['A'], 'confidence': x_positions, 
              'metrics': {'Class-wise ECE': np.sum(weight_options['A']*np.abs(confidences_bins['A']-accuracy['A'])), 
                         'Accuracy': (weight_options['A'] * accuracy['A']).sum()},
              'colors': ['#e6f7e5','#eef9ee','#ecf7ea','#e3f4e0','#daf0d6','#d1eccc','#c7e8c1','#bee5b7','#b5e1ad','#afd9a7']},
        'B': {'accuracy': accuracy['B'], 'confidence': x_positions, 
              'metrics': {'Class-wise ECE': np.sum(weight_options['B']*np.abs(confidences_bins['B']-accuracy['B'])), 
                         'Accuracy': (weight_options['B'] * accuracy['B']).sum()}, 
              'colors': ['#f2f8fc','#eff6fc','#f9fbfd','#f2f7fc','#ecf3fa','#e6eff8','#e0ecf7','#dae8f5','#d4e4f4','#cee0f2']},
        'C': {'accuracy': accuracy['C'], 'confidence': x_positions, 
              'metrics': {'Class-wise ECE': np.sum(weight_options['C']*np.abs(confidences_bins['C']-accuracy['C'])), 
                         'Accuracy': (weight_options['C'] * accuracy['C']).sum()}, 
              'colors': ['#fbe0cb','#fcede0','#ffeddf','#fee5cf','#fedcbe','#fed3ae','#feca9e','#fdc28e','#fdb97e','#fdb06e']},
        'D': {'accuracy': accuracy['D'], 'confidence': x_positions, 
              'metrics': {'Class-wise ECE': np.sum(weight_options['D']*np.abs(confidences_bins['D']-accuracy['D'])), 
                         'Accuracy': (weight_options['D'] * accuracy['D']).sum()}, 
              'colors': ['#f5f5fa','#f6f6fb','#f8f8fb','#f4f4f9','#f0f0f7','#ededf5','#e9e9f3','#e5e5f1','#e2e2ef','#dedeed']},
        'Total': {'accuracy': accuracy['Total'], 'confidence': x_positions, 
                 'metrics': {'Class-wise ECE': np.sum(weight_options['Total']*np.abs(confidences_bins['Total']-accuracy['Total'])), 
                            'Accuracy': (weight_options['Total'] * accuracy['Total']).sum()}, 
                 'colors': ['#f2fafa','#edf8f7','#e8f6f5','#e3f4f3','#e0f3f2','#dbf2f0','#d6f0ee','#d4efed','#cfedeb','#caebe9']}
    }   

    fig, axs = plt.subplots(2, 3, figsize=(11, 7))
    
    for ax, (key, value) in zip(axs.flatten(), data.items()):
        for i, (conf, acc) in enumerate(zip(value['confidence'], value['accuracy'])):
            alpha_idx = np.where(weight_rank_options[key]==i)[0][0]
            bar = ax.bar(conf, acc, width=0.1, color=value['colors'][alpha_idx], 
                        label=f'Proportion of {key}' if alpha_idx == 9 else "", 
                        align='edge', edgecolor='black', alpha=1)

            if underconfidence_bins[key][i] > 0:
                height = underconfidence_bins[key][i]
                bottom = confidences_bins[key][i]
                hatch_rect = Rectangle((conf, bottom), 0.1, height, 
                                     hatch='..', facecolor='none', edgecolor='gray', fill=False)
                ax.add_patch(hatch_rect)
            else:
                height = overconfidence_bins[key][i]
                top = acc
                hatch_rect = Rectangle((conf, top), 0.1, height, 
                                     hatch='//', facecolor='none', edgecolor='gray', fill=True, alpha=0.3)
                ax.add_patch(hatch_rect)

        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
        if key in ['A', 'D']:
            ax.set_ylabel('Bin-wise Option Proportion')
        else:
            ax.set_yticklabels([])
        
        if key in ['D', 'Total']:
            ax.set_xlabel('Confidence Level')
        else:
            ax.set_xticklabels([])
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

    for ax in axs.flatten()[5:]:
        fig.delaxes(ax)
    
    plt.tight_layout()
    plt.savefig('output_figures/cwece_plot.svg', format='svg', bbox_inches="tight")
    return fig, axs

def histogram_plotter_single(TP, Total, confidence_bins):
    # Calculate accuracy and confidence bins
    accuracy = np.array(torch.div(TP, Total).nan_to_num())
    confidences_bins = np.array(confidence_bins)
    underconfidence = np.maximum(0, accuracy - confidences_bins)
    overconfidence = np.maximum(0, confidences_bins - accuracy)
    weights = np.array(Total / Total.sum())
    weight_ranks = np.argsort(weights)

    # Define x positions for the bins
    x_positions = np.linspace(0.0, 0.9, 10)

    # Define bar colors based on weights
    colors = ['#e6f7e5', '#eef9ee', '#ecf7ea', '#e3f4e0', '#daf0d6',
              '#d1eccc', '#c7e8c1', '#bee5b7', '#b5e1ad', '#afd9a7']

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot each bin with accuracy, underconfidence, and overconfidence
    for i, (conf, acc) in enumerate(zip(x_positions, accuracy)):
        # Use the rank of weights to determine the color
        alpha_idx = np.where(weight_ranks == i)[0][0]
        bar = ax.bar(conf, acc, width=0.1, color=colors[alpha_idx],
                     label='Accuracy' if i == 0 else "", align='edge', edgecolor='black', alpha=1)

        # Add underconfidence or overconfidence patches
        if underconfidence[i] > 0:
            height = underconfidence[i]
            bottom = confidences_bins[i]
            hatch_rect = Rectangle((conf, bottom), 0.1, height, hatch='..',
                                    facecolor='none', edgecolor='gray', fill=False)
            ax.add_patch(hatch_rect)
        else:
            height = overconfidence[i]
            top = acc
            hatch_rect = Rectangle((conf, top), 0.1, height, hatch='//',
                                    facecolor='none', edgecolor='gray', fill=True, alpha=0.3)
            ax.add_patch(hatch_rect)

    # Add a perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

    # Configure axis labels and title
    ax.set_title('Reliability Diagram')
    ax.set_xlabel('Confidence Level')
    ax.set_ylabel('Accuracy')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left')

    # Save the plot
    plt.savefig('output_figures/ece_plot.svg', format='svg', bbox_inches="tight")
    plt.show()

