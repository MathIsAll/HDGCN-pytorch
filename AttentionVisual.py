import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pytorch_pretrained_bert import BertModel
from dataset import Tokenizer

# bert = BertModel.from_pretrained("bert/uncased.tar.gz")
# tokenizer = Tokenizer(max_seq_len=10, bert_vocab_path="bert/vocab.txt")
cmap = sns.color_palette('Paired', 16)
sns.set(font_scale=1.)


# sns.color_palette('Paired', 16)


def forward_heatmap_attention(attentions, sentence, heads, layer, batch):
    # tokens = tokenizer.tokenizer.convert_ids_to_tokens(sentence)
    hypernodes = ['h1', 'h2', 'h3', 'h4', 'h5']
    fig = plt.figure(figsize=(17.5, 2.5))
    axes = []
    if heads == 1:
        axes.append(fig.add_subplot(1, 1, 1))
        axes[-1].set_title(f"Head {1}")
        sns.heatmap(attentions.transpose(1, 0), vmax=1, vmin=0, yticklabels=hypernodes, xticklabels=sentence, annot=True,
                    ax=axes[-1],
                    cbar=False, linewidths=1, fmt='04.2f', cmap=cmap)
    else:
        for i_head in range(heads):
            axes.append(fig.add_subplot(1, heads, 1 + i_head))
            axes[-1].set_title(f"Head {1 + i_head}")
            sns.heatmap(attentions[i_head], vmax=1, vmin=0, yticklabels=tokens, xticklabels=tokens, annot=True,
                        ax=axes[-1],
                        cbar=False, linewidths=1, fmt='04.2f', cmap=cmap)
    plt.savefig("batch_{}_layer_{}_forward_attn.jpg".format(batch, layer))
    plt.close()


def backward_heatmap_attention(attentions, sentence, heads, layer, batch):
    # tokens = tokenizer.tokenizer.convert_ids_to_tokens(sentence)
    hypernodes = ['h1', 'h2', 'h3', 'h4', 'h5']
    fig = plt.figure(figsize=(17.5, 2.5))
    axes = []
    if heads == 1:
        axes.append(fig.add_subplot(1, 1, 1))
        axes[-1].set_title(f"Head {1}")
        sns.heatmap(attentions, vmax=1, vmin=0, yticklabels=hypernodes, xticklabels=sentence, annot=True,
                    ax=axes[-1],
                    cbar=False, linewidths=1, fmt='04.2f', cmap=cmap)
    else:
        for i_head in range(heads):
            axes.append(fig.add_subplot(1, heads, 1 + i_head))
            axes[-1].set_title(f"Head {1 + i_head}")
            sns.heatmap(attentions[i_head], vmax=1, vmin=0, yticklabels=tokens, xticklabels=tokens, annot=True,
                        ax=axes[-1],
                        cbar=False, linewidths=1, fmt='04.2f', cmap=cmap)
    plt.savefig("batch_{}_layer_{}_backward.jpg".format(batch, layer))
    plt.close()


def adj_heatmap_attention(attentions, sentence, heads, layer, batch):
    # tokens = tokenizer.tokenizer.convert_ids_to_tokens(sentence)
    tokens = ["_" for i in range(35)]
    # x_tokens = ['N1', 'N2', 'N3', 'N4', 'N5']
    fig = plt.figure(figsize=(17.5, 17.5))
    axes = []
    if heads == 1:
        axes.append(fig.add_subplot(1, 1, 1))
        axes[-1].set_title(f"Head {1}")
        sns.heatmap(attentions, vmax=1, vmin=0, yticklabels=sentence, xticklabels=sentence, annot=True,
                    ax=axes[-1],
                    cbar=False, linewidths=1, fmt='04.2f', cmap=cmap)
    else:
        for i_head in range(heads):
            axes.append(fig.add_subplot(1, heads, 1 + i_head))
            axes[-1].set_title(f"Head {1 + i_head}")
            sns.heatmap(attentions[i_head], vmax=1, vmin=0, yticklabels=tokens, xticklabels=tokens, annot=True,
                        ax=axes[-1],
                        cbar=False, linewidths=1, fmt='04.2f', cmap=cmap)
    plt.savefig("batch_{}_layer_{}_adj.jpg".format(batch, layer))
    plt.close()


def heatmap_attention(attentions, sentence, heads, layer, batch):
    # tokens = tokenizer.tokenizer.convert_ids_to_tokens(sentence)
    tokens = ["_" for i in range(35)]
    # x_tokens = ['N1', 'N2', 'N3', 'N4', 'N5']
    fig = plt.figure(figsize=(17.5, 17.5))
    axes = []
    if heads == 1:
        axes.append(fig.add_subplot(1, 1, 1))
        axes[-1].set_title(f"Head {1}")
        sns.heatmap(attentions, vmax=1, vmin=0, yticklabels=sentence, xticklabels=sentence, annot=True,
                    ax=axes[-1],
                    cbar=False, linewidths=1, fmt='04.2f', cmap=cmap)
    else:
        for i_head in range(heads):
            axes.append(fig.add_subplot(1, heads, 1 + i_head))
            axes[-1].set_title(f"Head {1 + i_head}")
            sns.heatmap(attentions[i_head], vmax=1, vmin=0, yticklabels=tokens, xticklabels=tokens, annot=True,
                        ax=axes[-1],
                        cbar=False, linewidths=1, fmt='04.2f', cmap=cmap)
    plt.savefig("batch_{}_layer_{}_dot_attn.jpg".format(batch, layer))
    plt.close()


def bipartite_attention(attentions, sentence, heads):
    tokens = tokenizer.tokenizer.convert_ids_to_tokens(sentence)
    fig = plt.figure(figsize=(10, 10))
    axes = []
    attns = attentions
    axes.append(fig.add_subplot(1, 1, 1))
    axes[-1].set_title(f'Adjacency Matrix')
    G = nx.DiGraph()
    top_node_labels, bottom_node_labels = zip(*tuple((f'{i} {tk}', f'{tk} {i}') for i, tk in enumerate(tokens)))
    G.add_nodes_from(top_node_labels, bipartite=0)
    G.add_nodes_from(bottom_node_labels, bipartite=1)
    G.add_weighted_edges_from([(tnl, bnl, weight) for tnl, row in zip(top_node_labels, attns) for bnl, weight in
                               zip(bottom_node_labels, row)])
    edge_ws = np.array([G[u][v]['weight'] for u, v in G.edges()])
    nx.draw_networkx(G, pos={
        **{tnl: (0, -i) for i, tnl in enumerate(top_node_labels)},
        **{bnl: (1, -j) for j, bnl in enumerate(bottom_node_labels)},
    }, width=edge_ws * 10, edge_color=edge_ws, edge_cmap=plt.cm.Accent, style='dotted', edge_vmin=0, edge_vmax=1,
                     min_source_margin=20, min_target_margin=20, node_size=0, font_size=10, arrowsize=20,
                     ax=axes[-1])
    plt.show()


def forward_bipartite_attention(attentions, sentence, heads):
    tokens = tokenizer.tokenizer.convert_ids_to_tokens(sentence)
    bottom_tokens = ['N1', 'N2', 'N3', 'N4', 'N5']
    fig = plt.figure(figsize=(10, 10))
    axes = []
    for i_head in range(heads):
        attns = attentions[i_head]
        axes.append(fig.add_subplot(1, heads, 1 + i_head))
        axes[-1].set_title(f'Head {1 + i_head}')
        G = nx.DiGraph()
        top_node_labels = tuple(tokens)
        bottom_node_labels = tuple(bottom_tokens)
        # top_node_labels, bottom_node_labels = zip(*tuple((f'{i} {tok}', f'{btok} {j}') for i, tok in enumerate(tokens) for j, btok in enumerate(bottom_tokens)))
        G.add_nodes_from(top_node_labels, bipartite=0)
        G.add_nodes_from(bottom_node_labels, bipartite=1)
        G.add_weighted_edges_from([(tnl, bnl, weight) for tnl, row in zip(top_node_labels, attns) for bnl, weight in
                                   zip(bottom_node_labels, row)])
        edge_ws = np.array([G[u][v]['weight'] for u, v in G.edges()])
        nx.draw_networkx(G, pos={
            **{tnl: (0, -i) for i, tnl in enumerate(top_node_labels)},
            **{bnl: (1, -j) for j, bnl in enumerate(bottom_node_labels)},
        }, width=edge_ws * 10, edge_color=edge_ws, edge_cmap=plt.cm.Accent, style='dotted', edge_vmin=0, edge_vmax=1,
                         min_source_margin=20, min_target_margin=20, node_size=0, font_size=10, arrowsize=20,
                         ax=axes[-1])
    plt.show()


def back_bipartite_attention(attentions, sentence, heads):
    top_tokens = ['N1', 'N2', 'N3', 'N4', 'N5']
    tokens = tokenizer.tokenizer.convert_ids_to_tokens(sentence)
    fig = plt.figure(figsize=(10, 10))
    axes = []
    for i_head in range(heads):
        attns = attentions[i_head]
        axes.append(fig.add_subplot(1, heads, 1 + i_head))
        axes[-1].set_title(f'Head {1 + i_head}')
        G = nx.DiGraph()
        top_node_labels = tuple(top_tokens)
        bottom_node_labels = tuple(tokens)
        # top_node_labels, bottom_node_labels = zip(*tuple((f'{i} {tok}', f'{btok} {j}') for i, tok in enumerate(tokens) for j, btok in enumerate(bottom_tokens)))
        G.add_nodes_from(top_node_labels, bipartite=0)
        G.add_nodes_from(bottom_node_labels, bipartite=1)
        G.add_weighted_edges_from([(tnl, bnl, weight) for tnl, row in zip(top_node_labels, attns) for bnl, weight in
                                   zip(bottom_node_labels, row)])
        edge_ws = np.array([G[u][v]['weight'] for u, v in G.edges()])
        nx.draw_networkx(G, pos={
            **{tnl: (0, -i) for i, tnl in enumerate(top_node_labels)},
            **{bnl: (1, -j) for j, bnl in enumerate(bottom_node_labels)},
        }, width=edge_ws * 10, edge_color=edge_ws, edge_cmap=plt.cm.Accent, style='dotted', edge_vmin=0, edge_vmax=1,
                         min_source_margin=20, min_target_margin=20, node_size=0, font_size=10, arrowsize=20,
                         ax=axes[-1])
    plt.show()
