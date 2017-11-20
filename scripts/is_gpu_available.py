"""
Want to know if the machine you're running on has been properly configured for
running tensorflow on a GPU? Run this script to find out. If it prints out the
following matrix:

    [[ 22.  28.]
     [ 49.  64.]]

then GPU support is enabled. If it prints out a super long stacktrace, then GPU
support is not enabled. This script was stolen from [1].

[1]: https://stackoverflow.com/a/43703735
"""

import tensorflow as tf

def _main():
    with tf.device('/gpu:0'):
        # pylint: disable=invalid-name
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)

    with tf.Session() as sess:
        print(sess.run(c))

if __name__ == "__main__":
    _main()
