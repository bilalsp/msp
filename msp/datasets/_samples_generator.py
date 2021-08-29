"""
Generate samples of synthetic data set for MSP problem.
"""
from typing import Tuple
import tensorflow as tf

from msp.graphs import MSPSparseGraph

    
def make_sparse_data(n_samples: int, 
                     batch_size: int = 1, 
                     n_node_feat: int = 5, 
                     n_edge_feat: int = 3, 
                     msp_size: Tuple[int, int] = None, 
                     msp_rand_size: Tuple[Tuple[int, int], Tuple[int, int]] = None, 
                     is_machine_idle: bool = True, 
                     seed=None) -> tf.data.Dataset:
    """Generate samples of synthetic data set for MSP problem.

    Args:
        n_samples: number of instances
        batch_size: batch_size is only used in conjunction with msp_rand_size
        n_node_feat: number of node features
        n_edge_feat: number of edge features
        msp_size: size of msp instance (number of jobs, number of machines)
        msp_rand_size: range for number of jobs and machines
        is_machine_idle: if false all machines will be connected in a graph 
        seed: seed to reproduce same data

    Returns:
        tensorflow dataset
    """
    assert not (msp_size is None and msp_rand_size is None), \
           "Either msp_size or msp_rand_size should be passed."

    # set random generator
    if seed:
        rand_gen = tf.random.experimental.Generator.from_seed(seed, alg="philox")
    else:
        rand_gen = tf.random.Generator.from_non_deterministic_state()

    def _generator(n_samples, batch_size):

        if msp_size:
            n_jobs, n_machines = msp_size

        for i in range(n_samples):
            if i % batch_size == 0 and msp_size is None:
                jobs_range, machines_range = msp_rand_size
                n_jobs = rand_gen.uniform(
                    shape=(), minval=jobs_range[0], 
                    maxval=jobs_range[1], dtype=tf.int32)
                n_machines = rand_gen.uniform(
                    shape=(), minval=machines_range[0], 
                    maxval=machines_range[1], dtype=tf.int32)

            n_nodes = n_jobs + n_machines

            node_type = tf.concat(
                [tf.ones([n_jobs, 1], tf.float32),
                tf.zeros([n_machines, 1], tf.float32)], axis = 0)

            # 1)) node feature tensor (n_nodes x n_node_feat)
            node_feat = tf.concat([
                tf.multiply(
                    rand_gen.uniform([n_nodes, n_node_feat - 1]),
                    tf.broadcast_to(node_type, [n_nodes, n_node_feat - 1]) 
                ),
                node_type
            ], axis = 1)

            # 2)) job assignment tensor (n_nodes x n_machines)
            job_assignment = rand_gen.uniform(shape=[n_jobs, n_machines])<=0.5
            job_assignment = tf.cast(job_assignment, dtype=tf.float32)

            # each job should be assigned to atleast one machine 
            mask_col = tf.broadcast_to(
                tf.equal(
                    tf.reduce_sum(job_assignment, axis=1, keepdims=True), 0),
                job_assignment.shape
            )
            mask_col = tf.cast(mask_col, dtype=tf.float32)
            job_assignment = tf.add(job_assignment, mask_col)

            # each machine has atleast one job assignment
            mask_row = tf.broadcast_to(
                tf.equal(
                    tf.reduce_sum(job_assignment, axis=0, keepdims=True), 0),
                job_assignment.shape
            )
            mask_row = tf.cast(mask_row, dtype=tf.float32)
            job_assignment = tf.add(job_assignment, mask_row)

            job_assignment = tf.concat(
                [job_assignment, tf.zeros([n_machines, n_machines], tf.float32)], axis = 0)            

            # 3)) adjacency tensor (n_nodes x n_nodes)
            flip_node_type = 1 - node_type

            # nodes i and j are jobs
            cond_1 = lambda : tf.cast(
                tf.equal(
                    node_type + tf.transpose(node_type), 2),
                dtype = tf.float32
            )

            # nodes i and j either both are machines or jobs.
            cond_2 = lambda : tf.cast(
                tf.not_equal(
                    flip_node_type + tf.transpose(flip_node_type), 1),
                dtype = tf.float32
            )

            cond_3 = tf.cast(
                tf.equal(
                    job_assignment @ (tf.transpose(job_assignment)), 0),
                dtype = tf.float32
            )

            adj_matrix = 1 - (tf.cond(is_machine_idle, cond_1, cond_2)  * cond_3)
            
            # no self loop
            adj_matrix = tf.linalg.set_diag(adj_matrix, tf.zeros((n_nodes,)))

            # 4)) edge feature tensor (n_nodes x n_nodes x n_edge_feat)

            # node i and j, none them is job
            edge_type_1 = lambda: tf.cast(
                tf.not_equal(
                    node_type + tf.transpose(node_type), 2),
                dtype = tf.float32
            )
            
            # node i and j, either of them is job
            edge_type_2 = lambda: tf.cast(
                tf.equal(
                    flip_node_type + tf.transpose(flip_node_type), 1),
                dtype = tf.float32
            )

            edge_type = tf.cond(is_machine_idle, edge_type_1, edge_type_2)
            flip_edge_type = 1 - edge_type

            # upper trianguler matrix
            edge_feat = tf.linalg.band_part(
                rand_gen.uniform((n_edge_feat - 1, n_nodes, n_nodes)), 0, -1)
            # symmetric matrix
            edge_feat = edge_feat + tf.transpose(edge_feat, perm=[0,2,1])

            edge_feat = tf.concat(
                [edge_feat * flip_edge_type, tf.expand_dims(edge_type, axis=0)], axis=0)
            edge_feat = tf.transpose(edge_feat * adj_matrix)

            yield MSPSparseGraph(adj_matrix, node_feat, edge_feat, job_assignment)

    output_signature = MSPSparseGraph(
        adj_matrix = tf.TensorSpec(shape=None, dtype=tf.float32), 
        node_features = tf.TensorSpec(shape=None, dtype=tf.float32), 
        edge_features = tf.TensorSpec(shape=None, dtype=tf.float32), 
        job_assignment = tf.TensorSpec(shape=None, dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(_generator, args=(n_samples,batch_size),
                                        output_signature=output_signature)
    return dataset
