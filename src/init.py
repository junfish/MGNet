import tensorflow as tf

initializer = tf.initializers.glorot_uniform()

def zeros(shape, name=None):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name, trainable = True, dtype=tf.float32)

def get_weight( shape , name ='None' ):
    # return tf.Variable(  tf.random.truncated_normal(shape,stddev=0.1) , name= name , trainable = True , dtype=tf.float32 )
    return tf.Variable(  initializer(shape) , name = name, trainable=True , dtype=tf.float32 )

def initialize_bias_weights(shapes):
  weights = []
  for i in range( len( shapes ) ):
    weights.append( get_weight( shapes[ i ] , 'weight{}'.format( i )) )

  bias = []
  for i in range(len(shapes)):
    bias_shape = shapes[i][1]
    bias_i = zeros(bias_shape, 'bias{}'.format( i ))
    bias.append(bias_i)

  return weights, bias

def initialize_shape(num_layers, Input_dim, Output_dim, M, process):
  shapes = []

  for i in range(num_layers):
    if(i == 0):
      shapes_i = [Input_dim, Output_dim[0]]
    elif(i == num_layers - 1):
      if(process == 'concat'):
        shapes_i = [M * Output_dim[i-1], Output_dim[i]]
      else:
        shapes_i = [Output_dim[i-1], Output_dim[i]]

    else:
      shapes_i = [Output_dim[i-1], Output_dim[i]]

    shapes.append(shapes_i)

  return shapes

if __name__ == "__main__":
    weights, bias = initialize_bias_weights(shapes)
    num_layers = 2
    Input_dim = 80
    M = 30
    process = 'concat'
    Output_dim = [60,2]
    shapes = initialize_shape(num_layers, Input_dim, Output_dim, M, process)
    print(shapes)

