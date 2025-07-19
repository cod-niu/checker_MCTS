# import tensorflow as tf
# print(tf.__version__)
# print(tf.config.list_physical_devices('GPU'))


from keras.models import load_model

model = load_model('data/model/Checkers_Model10_12-Feb-2021(14:50:36).h5')
# model.summary()  # 打印结构和每层参数量

# # 统计总参数量
# total_params = model.count_params()
# print("Total parameters:", total_params)

# # 估算参数占用显存（float32每参数4字节）
# param_mem_MB = total_params * 4 / 1024 / 1024
# print("Parameter memory (MB):", param_mem_MB)


import time
import tensorflow as tf

# 预热
model.predict(dummy_input)

# 等待显存稳定
time.sleep(2)

# 记录推理前显存
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        info = tf.config.experimental.get_memory_info(gpu.name)
        print(f"Before inference: {info}")

# 真正推理
output = model.predict(real_input)

# 记录推理后显存
if gpus:
    for gpu in gpus:
        info = tf.config.experimental.get_memory_info(gpu.name)
        print(f"After inference: {info}")