## Download
https://drive.google.com/drive/folders/1BJxNYjArzvw7LtrRGCLeBHyJxB9no5tJ?usp=sharing

## Run
```
adb devices
adb push benchmark_model /data/local/tmp
adb shell chmod a+x /data/local/tmp/benchmark_model
adb push your_model.tflite /data/local/tmp
adb shell taskset 70 /data/local/tmp/benchmark_model --graph=/data/local/tmp/your_model.tflite --num_thread=1 
```

## Reference
https://www.tensorflow.org/lite/performance/measurement
