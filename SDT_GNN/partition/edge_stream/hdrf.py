import numpy as np
import pandas as pd
import random
import json
import pickle
import os
import csv
import gzip
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
# from memory_profiler import profile
from SDT_GNN.partition.partitioner import Partitioner

class HDRF(Partitioner):
    """
    An implementation of High Degree (are) Replicated First (HDRF) partitioning method.
    For details about the algorithm see this paper:
    "HDRF: Stream-based Partitioning for Power-Law Graphs"

    Args:
        dataset (str): Dataset name.
        path (str): Dataset root path.
        output_path (str): Output path.
        number_partition (int): Number of partitions.
        Lambda (float): A parameter controls load imbalance.
        K (int): Number of hops of neighbor maintained after partitioning. Default is 1.
        seed (int): Random seed. Default is 42.
        partition_features_file (bool): Partition the features file if True,
        print_partition_statistics (bool): Print out the statistics of the partitioned graph if True.
    """

    def __init__(self, 
                 dataset: str = None, 
                 multilabel: bool=False, 
                 path: str = None, 
                 output_path: str = None, 
                 number_partition: int = 4, 
                 Lambda: float = 1.0, 
                 K: int = 1, 
                 seed: int = 42,
                 partition_features_file: bool = True,
                 print_partition_statistics: bool = True):
        super().__init__()

        self.dataset = dataset
        self.multilabel = multilabel
        self.path = path
        self.output_path = output_path
        self.number_partition = number_partition
        self.Lambda = Lambda
        self.K = K
        self.seed = seed
        self._set_seed()
        self.node_degree = dict()

        self.partition_features_file = partition_features_file
        self.print_partition_statistics = print_partition_statistics
        
        self.epsilon = 1
        self.edge_load = [0 for _ in range(self.number_partition)]
        self.vertex_partition_matrix  = defaultdict(list)
    
    
    def update_vertex_partition_matrix(self, i,j, max_p):
        self.vertex_partition_matrix[i][max_p] = True
        self.vertex_partition_matrix[j][max_p] = True


    def partition(self):
        """Partition a graph."""

        self.v2p = defaultdict(int)
        
        for i in range(self.number_partition):
            with open(os.path.join(self.output_path, 'partition_' + str(i) + '.txt'), 'w') as f:
                pass

        self.file = open(self.path + self.dataset + '/edge_list.csv', 'r')
        
        file_objects = []
        for i in range(self.number_partition):
            filename = os.path.join(self.output_path, 'partition_' + str(i) + '.txt')
            files = open(filename, 'a', newline='')
            # Append the file object to the list
            file_objects.append(files)
        
        reader = csv.reader(self.file, delimiter=',')
        
        for number_edges, line in enumerate(reader):
            i, j = int(line[0]), int(line[1])
                
            if i in self.node_degree.keys():
                self.node_degree[i] += 1
            else:
                self.node_degree[i] = 1
                for _ in range(self.number_partition):
                    self.vertex_partition_matrix[int(i)].append(False)

            if j in self.node_degree.keys():
                self.node_degree[j] += 1
            else:
                self.node_degree[j] = 1
                for _ in range(self.number_partition):
                    self.vertex_partition_matrix[int(j)].append(False)

            d_i = self.node_degree[i]
            d_j = self.node_degree[j]

            maxsize = max(self.edge_load)
            minsize = min(self.edge_load)

            C_hdrf = [0 for _ in range(self.number_partition)]
            for p in range(self.number_partition):
                g_i = 0
                g_j = 0
                if self.vertex_partition_matrix[i][p]:
                    g_i = d_i/(d_i + d_j)
                    g_i = 1 + (1 - g_i)
                if self.vertex_partition_matrix[j][p]:
                    g_j = d_j/(d_i + d_j)
                    g_j = 1 + (1 - g_j)

                load = self.edge_load[p]
                C_rep = g_i + g_j
                C_bal = self.Lambda * (maxsize - load)/(self.epsilon + maxsize - minsize)
                C_hdrf[p] = C_rep + C_bal
            partition_id = np.argmax(C_hdrf)
            self.edge_load[partition_id] += 1
            self.update_vertex_partition_matrix(i, j, partition_id)

            self.v2p[j] = partition_id
            
            writer = csv.writer(file_objects[partition_id], delimiter=' ')
            writer.writerow(line)
        
        self.number_edges = number_edges + 1
        self.number_nodes = max(self.v2p) + 1
        self.file.close()
        print('Number of nodes: ', self.number_nodes)
        print('Number of edges: ', self.number_edges)
        
        with open(self.output_path + 'partition' + '.json', 'wb') as json_file:
            pickle.dump(self.v2p, json_file)
        
          
        if self.K == 0:
            pass
            
        elif self.K == 1:
            self.file = open(self.path + self.dataset + '/edge_list.csv', 'r')
            reader = csv.reader(self.file, delimiter=',')
            for _, line in enumerate(reader):
                i, j = int(line[0]), int(line[1])
                partition_id = self.v2p[j]
                
                writer = csv.writer(file_objects[partition_id], delimiter=' ')
                writer.writerow(line)
            self.file.close()
            
        else:
            for l in range(self.number_partition):
                node_set = [k for k,v in self.v2p.items() if v == l]
                new_node_set = set()
                for _ in range(self.K):
                    self.file = open(self.path + self.dataset + '/edge_list.csv', 'r')
                    reader = csv.reader(self.file, delimiter=',')
                    for _, line in enumerate(reader):
                        i, j = int(line[0]), int(line[1])
                        if j in node_set:
                            writer = csv.writer(file_objects[l], delimiter=' ')
                            writer.writerow(line)
                            new_node_set.add(i)
                    node_set = new_node_set
                    self.file.close()

        for files in file_objects:
            files.close()

