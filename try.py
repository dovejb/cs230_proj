import tensorflow as tf

a = tf.Variable([0.1,0.2,0.3,0.4])
print("argmax", tf.math.argmax(a))
print("reduce_max", tf.math.reduce_max(a))
print(a > 0.25)

print("max", max(1,2))