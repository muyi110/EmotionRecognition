from tensorflow.python import pywrap_tensorflow
import os
import numpy as np
model_dir = "./"
checkpoint_path = os.path.join(model_dir, "train_model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
total_parameters = 0
for key in var_to_shape_map:#list the keys of the model
    shape = np.shape(reader.get_tensor(key))  #get the shape of the tensor in the model
    shape = list(shape)
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim
    total_parameters += variable_parameters

print(total_parameters)
