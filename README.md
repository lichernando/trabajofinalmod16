## Comandos utilizados

### Instalación de dependencias
```bash
# Crear el ambiente
conda create -n mlops python=3.9 ipykernel
# Activar el ambiente
conda activate mlops
# Agregar un entorno recién creado al notebook como kernel
python -m ipykernel install --user --name=deployment
# Instalar dependencias
pip install notebook numpy pandas scikit-learn imblearn matplotlib mlflow dvc dvc-gdrive opencv-python tensorflow tensorflow_datasets

# Inicializar Jupyter
jupyter notebook

```

### Crear un contenedor


```bash

docker build -t mlops:v1 . 

docker run -dp 8082:5002 -ti --name despliegue mlops:v1

curl -F "file=@79df86f4ad778c926f80450c75827e.png" http://localhost:8082/predict
```