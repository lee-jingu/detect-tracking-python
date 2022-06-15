# inference
다양한 detector와 tracker를 연결하는 inference module

## Docker Setup

```bash
bash setup/docker_setup.sh
```

## Docker Image Setup

```bash
bash setup/docker_build.sh
```


## Install TensorRT

```bash
python setup/setup_tensorrt.py --install
```

## Run torch to onnx

```bash
python torch_to_onnx.py
```


## Run Demo

```bash
python demo.py
```
