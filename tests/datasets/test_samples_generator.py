"""
The :mod:`tests.datasets.test_samples_generator` module tests the 
`msp.datasets._samples_generator` module.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from absl.testing import parameterized
from parameterized import parameterized_class
from keras import keras_parameterized
import tensorflow as tf

from msp.datasets import make_sparse_data


@parameterized_class([
    {"is_machine_idle": True},
    {"is_machine_idle": False}])
class TestSuiteAdjacencyMatrix(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        if hasattr(cls, 'is_machine_idle'):
            msp_rand_size = ((20, 50), (5, 15))
            n_instances, batch_size = 3200, 32
            dataset = make_sparse_data(n_instances, 
                                       batch_size, 
                                       msp_rand_size=msp_rand_size, 
                                       is_machine_idle=cls.is_machine_idle, 
                                       seed=2021)
            cls.batch_dataset = dataset.batch(batch_size)

    def setUp(self):
        """Each testcase run on different edge feature and adjacency matrix"""
        if hasattr(self, 'is_machine_idle'):
            batch = list(self.batch_dataset.take(1))[0]
            self.msp_size = batch.msp_size 
            self.adj_matrix = batch.adj_matrix

    def test_self_loop(self):
        """AdjacencyMatrix of MSP graph does not have self loop."""
        adj_matrix = self.adj_matrix
        n_jobs, n_machines = self.msp_size

        # no self loop
        diag = tf.linalg.diag_part(adj_matrix) 
        expected_diag = tf.zeros(diag.shape)

        self.assertAllClose(diag, expected_diag)

    def test_machine_edge(self):
        """Edge for machine-machine pair and machine-job pair."""
        adj_matrix = self.adj_matrix
        n_jobs, n_machines = self.msp_size
                 
        # there is always an edge between machine and job
        machine_job = adj_matrix[:, n_jobs:, :n_jobs]
        expected_machine_job = tf.ones(machine_job.shape) 

        if self.is_machine_idle:
            # there is always an edge between machine and machine
            machine_machine = adj_matrix[:, n_jobs:, n_jobs:]
            expected_machine_machine = tf.ones(machine_machine.shape)
            expected_machine_machine = tf.linalg.set_diag(
                expected_machine_machine, 
                tf.zeros((machine_machine.shape[0], machine_machine.shape[1]))
            ) 
        else:
            # there is always an edge between machine and machine
            machine_machine = adj_matrix[:, n_jobs:, n_jobs:]
            expected_machine_machine = tf.zeros(machine_machine.shape)

        self.assertAllClose(machine_machine, expected_machine_machine)
        self.assertAllClose(machine_job, expected_machine_job) 


@parameterized_class(
    [
        {"is_machine_idle": True},
        {"is_machine_idle": False}
    ]
)
class TestSuiteEdgeFeatureMatrix(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        if hasattr(cls, 'is_machine_idle'):
            msp_rand_size = ((20, 50), (5, 15))
            n_instances, batch_size = 3200, 32
            dataset = make_sparse_data(n_instances, 
                                       batch_size, 
                                       msp_rand_size=msp_rand_size, 
                                       is_machine_idle=cls.is_machine_idle, 
                                       seed=2021)
            cls.batch_dataset = dataset.batch(batch_size)

    def setUp(self):
        """Each testcase run on different edge feature and adjacency matrix"""
        if hasattr(self, 'is_machine_idle'):
            batch = list(self.batch_dataset.take(1))[0]
            self.msp_size = batch.msp_size 
            self.edge_features = batch.edge_features
            self.adj_matrix = batch.adj_matrix

    def test_no_edge_feature(self):
        """Edge feature for unconnected nodes"""    
        n_jobs, n_machines = self.msp_size 
        edge_features = self.edge_features
        adj_matrix = self.adj_matrix

        # if there is no edge between nodes then edge feature is a zero vector
        mask = 1 - adj_matrix
        zero_vectors = tf.boolean_mask(edge_features, mask)
        
        tf.assert_equal(zero_vectors, tf.constant(0, dtype=tf.float32))

    def test_machine_job_feature(self):
        """Edge feature for machine-job and machine-machine edges"""
        n_jobs, n_machines = self.msp_size 
        edge_features = self.edge_features
        adj_matrix = self.adj_matrix

        # edge feature for (machine-job and machine-machine) 
        mask = adj_matrix[:, n_jobs:, :]
        edge_machine_job = tf.boolean_mask(edge_features[:, n_jobs:, :, :], mask)
        expected_edge_machine_job = tf.concat([
            tf.zeros((edge_machine_job.shape[0], edge_machine_job.shape[1] - 1)),
            tf.ones((edge_machine_job.shape[0], 1))
        ], axis=-1)

        self.assertAllClose(edge_machine_job, expected_edge_machine_job)


class TestSuiteNodeFeatureMatrix(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        msp_rand_size = ((20, 50), (5, 15))
        n_instances, batch_size = 3200, 32
        dataset = make_sparse_data(n_instances, batch_size, 
                                   msp_rand_size=msp_rand_size, seed=2021)
        cls.batch_dataset = dataset.batch(batch_size)

    def setUp(self):
        """Each testcase run on different node feature matrix"""
        batch = list(self.batch_dataset.take(1))[0]
        self.msp_size = batch.msp_size 
        self.node_features = batch.node_features

    def test_machine_features(self):
        """Node feature for machine node"""
        n_jobs, n_machines = self.msp_size 
        node_features = self.node_features

        # Machine node does not have any feature
        machine_feature = node_features[:, n_jobs:, :]
        expected_machine_feature = tf.zeros(machine_feature.shape)

        self.assertAllClose(machine_feature, expected_machine_feature)

    def test_node_type(self):
        """Node type in graph"""
        n_jobs, n_machines = self.msp_size 
        node_features = self.node_features
        batch_size = node_features.shape[0]
        
        # job and machine are represented by is 0 and 1 resp.
        node_type_index = node_features.shape[-1] - 1
        node_type = tf.gather(node_features, indices=[node_type_index], axis=-1)
        
        total_jobs = tf.reduce_sum(node_type)
        expected_total_jobs = tf.constant(batch_size * n_jobs)

        self.assertAllClose(total_jobs, expected_total_jobs)


class TestSuiteJobAssignmentMatrix(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        msp_rand_size = ((20, 50), (5, 15))
        n_instances, batch_size = 3200, 32
        dataset = make_sparse_data(n_instances, batch_size, 
                                   msp_rand_size=msp_rand_size, seed=2021)
        cls.batch_dataset = dataset.batch(batch_size)

    def setUp(self):
        """Each testcase run on different job assignement matrix."""
        batch = list(self.batch_dataset.take(1))[0]
        self.msp_size = batch.msp_size 
        self.job_assignment = batch.job_assignment

    def test_job_assignment(self):
        """Test assignment of job to machine."""
        n_jobs, n_machines = self.msp_size 
        job_assignment = self.job_assignment

        # total number of jobs which can be assigned to each machine
        total_jobs = tf.reduce_sum(job_assignment, axis=-2)
        # total number of machines where each job can be assigned
        total_machines = tf.reduce_sum(job_assignment[:, :n_jobs, :], axis=-1)

        self.assertAllGreater(total_machines, 0)
        self.assertAllGreater(total_jobs, 0)

    def test_machine_to_machine(self):
        """Test machine to machine assignment."""
        n_jobs, n_machines = self.msp_size 
        job_assignment = self.job_assignment

        # there is no `machine to machine` assignment 
        machine_to_machine = job_assignment[:, n_jobs:, :]
        expected_machine_to_machine = tf.zeros(machine_to_machine.shape)

        self.assertAllClose(machine_to_machine, expected_machine_to_machine)


class TestSuiteRandomSeed(keras_parameterized.TestCase):

    @parameterized.named_parameters(
        (
            'fixed size MSP',
            {'msp_size': (40, 10)}
        ),
        (
            'Variable size MSP',
            {'msp_rand_size': ((20, 50), (5, 15)), 'batch_size': 8}
        )
    )
    def test_seed(self, kwargs):
        n_instances = 16
        ds1 = make_sparse_data(n_instances, **kwargs, seed=2021)
        ds2 = make_sparse_data(n_instances, **kwargs, seed=2021)

        for msp1, msp2 in zip(ds1, ds2):
            tf.assert_equal(msp1.adj_matrix, msp2.adj_matrix)
            tf.assert_equal(msp1.node_features, msp2.node_features)
            tf.assert_equal(msp1.edge_features, msp2.edge_features)
            tf.assert_equal(msp1.job_assignment, msp2.job_assignment)

    def test_no_seed(self):
        n_instances = 16
        ds1 = make_sparse_data(n_instances, msp_size=(40, 10), seed=None)
        ds2 = make_sparse_data(n_instances, msp_size=(40, 10), seed=None)

        for msp1, msp2 in zip(ds1, ds2):
            tf.assert_equal(
                tf.reduce_all(tf.equal(msp1.adj_matrix, msp2.adj_matrix)), 
                False
            )
            tf.assert_equal(
                tf.reduce_all(tf.equal(msp1.node_features, msp2.node_features)), 
                False
            )
            tf.assert_equal(
                tf.reduce_all(tf.equal(msp1.edge_features, msp2.edge_features)), 
                False
            )
            tf.assert_equal(
                tf.reduce_all(tf.equal(msp1.job_assignment, msp2.job_assignment)), 
                False
            )
