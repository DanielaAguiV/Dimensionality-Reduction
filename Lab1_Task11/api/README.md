# Prediction API

### Ejecución local

Inicialmente se deben de configurar las variables de entorno necesarias en el archivo **cfg.env**. Se ha desarrollado una API usando el modulo **FastAPI**, para correrla desde una máquina local, desde consola ejecutar el siguiente comando

    python<version> -m uvicorn app:app --env-file "cfg.env"

Este comando lanza la API de flask en el localhost a través del puerto 8000, cuenta con un endpoint, que se muestran a continuación 

- **"/uploadfile"**  se sube una imagen y retorna una predicción

Para acceder al swagger de la API, desde el navegador acceda a la dirección

    localhost:8000/docs


