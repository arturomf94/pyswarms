import test_framework as tf

test = tf.TestAlgorithm('search feasible region',
                        [tf.cops.C01(), tf.cops.C03()],
                        3,
                        [3, 5])

test.run()
