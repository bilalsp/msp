"""
The :mod:`mps.utils.objective` module defines `objective` function 
of MSP problem. 
"""
import tensorflow as tf


def compute_makespan(inputs, schedules):
    """Compute makespan for MSP problem."""
    num_job, num_machine = inputs.msp_size
    num_node = inputs.num_node
    batch_size = inputs.batch_size

    # processing time of jobs
    proc_time_index = 0
    proc_time = inputs.node_features[:, :, proc_time_index]

    # processing time of each job in a given schedule
    # B x V
    proc_time_schedule = tf.gather_nd(
        proc_time, tf.expand_dims(schedules[:,:,0], axis=-1), batch_dims=1)

    # setup time between the jobs
    setup_time_index = 0
    setup_time = inputs.edge_features[:, :, :, setup_time_index]

    # setup time between two consecutives jobs in a given schedule
    indices = tf.stack(
        [schedules[:,:,0], tf.roll(schedules[:,:,0], shift=-1, axis=-1)], axis=-1)
    # B x V
    setup_time_schedule = tf.gather_nd(setup_time, indices, batch_dims=1)

    schedules_with_setup_and_proc_time = tf.concat([
        tf.cast(schedules, dtype=tf.float32),
        tf.stack([proc_time_schedule, setup_time_schedule], axis=-1)
    ], axis=-1)
 
    # 
    shape_ = (batch_size, num_machine, num_node, 2)

    # setup time and processing time 
    setup_and_proc_time = tf.broadcast_to(
        schedules_with_setup_and_proc_time[:,tf.newaxis, :, -2:],
        shape = shape_
    )

    # machine id of job-to-machine assignment in schedule
    schedules_machine_idx = tf.broadcast_to(
        schedules[:, tf.newaxis, :, 1, tf.newaxis], 
        shape = shape_
    )

    # 
    res = tf.broadcast_to(
        tf.range(num_job, num_job + num_machine, dtype=tf.int64)\
            [tf.newaxis, :, tf.newaxis, tf.newaxis],
        shape = shape_
    )

    # processing time and setup time of all job assigned to each machine in the sytem
    res = tf.multiply(
        setup_and_proc_time,
        tf.cast(tf.equal(res, schedules_machine_idx), dtype=tf.float32)
    )
    
    # completion time of the last job over each machines in the system
    # B x num_machines
    comp_time = tf.reduce_sum(tf.reduce_sum(res, axis=-1), axis=-1)
    
    # completion time of the last job to leave the system
    # B x 1
    makespan = tf.reduce_max(comp_time, axis=-1, keepdims=True)

    return makespan
