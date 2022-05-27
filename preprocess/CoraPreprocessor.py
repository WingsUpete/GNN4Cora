import os
import sys
import requests
import tarfile
import shutil
import argparse
import json

import torch

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
sys.stderr.close()
sys.stderr = stderr

DATA_URL = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
TGZ_FN = 'cora.tgz'
CITES_DATA_FN = 'cora.cites'
CONTENT_DATA_FN = 'cora.content'

DATA_DIR_DEFAULT = '../data/cora/'


def get_num_edges(edges):
    cnt = 0
    for edge in edges:
        cnt += len(edge)
    return cnt


def tran_src_dst_list(edges):
    src_list, dst_list = [], []
    for i in range(len(edges)):
        cur_src_id = i
        for cur_dst_id in edges[i]:
            src_list, dst_list = form_edge(src_list, dst_list, src_id=cur_src_id, dst_id=cur_dst_id)
    return src_list, dst_list


def form_edge(src_list: list, dst_list: list, src_id: int, dst_id: int):
    """ Wraps the process to form an edge using the source and destination node lists. """
    src_list.append(src_id)
    dst_list.append(dst_id)
    return src_list, dst_list


class CoraPreProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not self.check_data_integrity():
            self.prepare_data()
        print('> [CoraPreprocessor:init] %s successfully initialized.' % self.__class__.__name__)

    def check_data_integrity(self):
        """ Checks data integrity, i.e., whether we have all the raw data needed. """
        return os.path.isdir(self.data_dir) and \
               os.path.isfile(os.path.join(self.data_dir, CITES_DATA_FN)) and \
               os.path.isfile(os.path.join(self.data_dir, CONTENT_DATA_FN))

    def prepare_data(self):
        """ Prepares the data if necessary: download, extract and reposition. """
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        tgz_path = os.path.join(self.data_dir, TGZ_FN)

        if os.path.isfile(tgz_path):
            print('> [CoraPreprocessor:prepare_data] Data already downloaded as %s' % tgz_path)
        else:
            print('> [CoraPreprocessor:prepare_data] Downloading data from %s to %s' % (DATA_URL, tgz_path))
            response = requests.get(DATA_URL, stream=True)
            if response.status_code == 200:
                with open(tgz_path, 'wb') as f:
                    f.write(response.raw.read())

        print('> [CoraPreprocessor:prepare_data] Extracting files...')
        with tarfile.open(tgz_path) as f:
            f.extractall(self.data_dir)
        temp_dir = os.path.join(self.data_dir, 'cora/')
        for file in os.listdir(temp_dir):
            shutil.move(os.path.join(temp_dir, file), self.data_dir)
        os.rmdir(temp_dir)

        print('> [CoraPreprocessor:prepare_data] Data is now ready on %s' % self.data_dir)

    def construct_graph(self, pub_dict: dict, save=True):
        """
        Constructs a DGLGraph which stores the graph structure, and the word feature vectors, labels for nodes
        :param pub_dict: a publication dictionary generated using "construct_pub_dict"
        :param save: specifies whether the constructed graph will be saved to local disk as "cora.dgl"
        :return: a DGLGraph constructed from pub_dict
        """
        print('> [CoraPreprocessor:construct_graph] Constructing DGLGraph...')
        graphs = [dgl.to_simple(dgl.graph(tran_src_dst_list(pub_dict['edges'][i]), num_nodes=len(pub_dict['nodes'])))
                  for i in range(len(pub_dict['edges']))]

        features = []
        labels = []
        for node in pub_dict['nodes']:
            features.append(node['feat'])
            labels.append(node['label'])
        features = torch.stack(features)
        labels = torch.Tensor(labels).long()
        for i in range(len(graphs)):
            graphs[i].ndata['feat'] = features
            graphs[i].ndata['label'] = labels

        if save:
            graph_path = os.path.join(self.data_dir, 'cora.dgl')
            dgl.save_graphs(graph_path, graphs)
            print('> [CoraPreprocessor:construct_graph] DGLGraph saved to %s' % graph_path)

        return graphs

    def construct_pub_dict(self, save_meta=True):
        """
        Generates publications dictionary according to the specified graph relationship.
        As for relationship, for "citing", if x cites y, then y --> x;
        for "cited", if y cites x, then y --> x;
        for "both", x <--> y as long as they have citing relationship in any direction.
        :param save_meta: specifies whether the meta data will be saved to local disk as "pub_dict.json"
        :return: a dictionary including the needed information (nodes, edges, node features, node labels)
        """
        # 1. Process contents: publication id, word feature vector, label
        label_set = set()
        node_id_counter = 0
        nodes = []
        pub_id2node_id = {}
        content_path = os.path.join(self.data_dir, CONTENT_DATA_FN)
        print('> [CoraPreprocessor:construct_pub_dict] Processing %s' % content_path)
        with open(content_path) as f:
            while True:
                line = f.readline().strip()
                if not line:
                    break
                vals = line.split('\t')

                # Process this line of data - publication/node
                cur_label = vals[-1]
                label_set.add(cur_label)
                cur_node: dict = {
                    'node_id': node_id_counter,
                    'pub_id': int(vals[0]),
                    'feat': torch.Tensor([float(val) for val in vals[1:-1]]),
                    'label_str': cur_label
                }
                nodes.append(cur_node)
                pub_id2node_id[cur_node['pub_id']] = cur_node['node_id']

                node_id_counter += 1
        # Provide numeric labels
        label_str2val = {}
        sorted_labels = sorted(list(label_set))
        label_val_counter = 0
        for cur_label in sorted_labels:
            label_str2val[cur_label] = label_val_counter
            label_val_counter += 1
        for node in nodes:
            node['label'] = label_str2val[node['label_str']]

        # process cites: citing relationship among publications
        citing_edges = [set() for _ in range(len(nodes))]
        cited_edges = [set() for _ in range(len(nodes))]
        both_edges = [set() for _ in range(len(nodes))]
        cites_path = os.path.join(self.data_dir, CITES_DATA_FN)
        print('> [CoraPreprocessor:construct_pub_dict] Processing %s' % cites_path)
        with open(cites_path) as f:
            while True:
                line = f.readline().strip()
                if not line:
                    break
                vals = line.split('\t')

                # Process this line of data - citing_relationship/edge
                cited_pub_id, citing_pub_id = int(vals[0]), int(vals[1])
                cited_node_id = pub_id2node_id[cited_pub_id]
                citing_node_id = pub_id2node_id[citing_pub_id]

                # Citing Relationship
                citing_edges[cited_node_id].add(citing_node_id)
                # Cited Relationship
                cited_edges[citing_node_id].add(cited_node_id)
                # Both Relationship
                both_edges[cited_node_id].add(citing_node_id)
                both_edges[citing_node_id].add(cited_node_id)

        print('> [CoraPreprocessor:construct_pub_dict] Collecting components to construct the publication dictionary...')
        pub_dict = {
            'nodes': nodes,
            'edges': [citing_edges, cited_edges, both_edges],
            'extra': {
                'label_map': label_str2val,
                'id_map_pub2node': pub_id2node_id
            }
        }
        meta = {
            'num_nodes': len(pub_dict['nodes']),
            'num_edges': [get_num_edges(edges) for edges in pub_dict['edges']],
            'num_feats': pub_dict['nodes'][-1]['feat'].shape[-1],
            'num_classes': len(pub_dict['extra']['label_map']),
            'label_map': pub_dict['extra']['label_map'],
            'relationship_map': {
                'citing': 0,
                'cited': 1,
                'both': 2
            },
            'id_map_pub2node': pub_id2node_id,
        }
        if save_meta:
            meta_path = os.path.join(self.data_dir, 'meta.json')
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            print('> [CoraPreprocessor:construct_pub_dict] Publication dictionary saved to %s' % meta_path)

        return pub_dict, meta


if __name__ == '__main__':
    """ 
        Usage Example:
        python CoraPreprocessor.py -dr ../data/cora/
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-dr', '--data_dir', type=str, default=DATA_DIR_DEFAULT, help='Root directory of the input data, default = {}'.format(DATA_DIR_DEFAULT))

    FLAGS, unparsed = parser.parse_known_args()

    cora_preproc = CoraPreProcessor(data_dir=FLAGS.data_dir)
    my_pub_dict, my_meta = cora_preproc.construct_pub_dict(save_meta=True)
    my_graphs = cora_preproc.construct_graph(my_pub_dict, save=True)
