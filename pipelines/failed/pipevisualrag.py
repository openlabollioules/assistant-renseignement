import os
import sys
import typing
from typing import List, Generator, Iterator

import requests
#import duckdb
import io
import uuid
import pandas as pd
import json
import shutil
import base64
import re


# Pour la classification
from ultralytics import YOLO

# Pour Ollama
from langchain_ollama import OllamaLLM

# Pour la création d'embeddings et l'indexation
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, Settings
from llama_index.core.embeddings.utils import resolve_embed_model

from llama_index.llms.ollama import Ollama
#from llama_index.embeddings import BaseEmbedding


from llama_index.embeddings.openai import OpenAIEmbedding

# Pour gérer l'exception de connexion
from urllib3.exceptions import NewConnectionError

# Pydantic & FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_ollama import OllamaLLM
#from llama_index.embeddings.ollama import OpenAIEmbedding

from utils.pipelines.main import get_last_user_message, get_last_assistant_message






class Pipeline:
    """
    Exemples d'initialisation de modèles Ollama + YOLO + Indexation.
    """

    class Valves(BaseModel):
        # Variables de configuration pour l'utilisation d'Ollama
        LLAMAINDEX_OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
        LLAMAINDEX_MODEL_NAME: str = "mistral-small:latest"
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str = "bge-m3:latest"

    def __init__(self):
        self.documents = None
        self.index = None

        # Création d'une instance de Valves (récupération config depuis environnements si besoin)
        self.valves = self.Valves(
            LLAMAINDEX_OLLAMA_BASE_URL=os.getenv("LLAMAINDEX_OLLAMA_BASE_URL",
                                                 "http://host.docker.internal:11434"),
            LLAMAINDEX_MODEL_NAME=os.getenv("LLAMAINDEX_MODEL_NAME",
                                            "mistral-small:latest"),
            LLAMAINDEX_EMBEDDING_MODEL_NAME=os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME",  "bge-m3:latest"),
        )

        # Modèle d'embeddings Ollama
        #self.embed_model = OllamaEmbedding(
        #    base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,  # Ollama server
        #    model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,                    # Embeddings model
        #)

        # Test de la connexion Ollama pour les embeddings
        #self.test_ollama_embeddings()

        print(f"Valves initialized: {self.valves}")
        try:
            # Modèle de raisonnement Ollama
            self.reasoning_model = OllamaLLM(
                model=self.valves.LLAMAINDEX_MODEL_NAME,
                base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            )

            # Modèle de classification YOLO
            self.model_classif = YOLO("/app/share/classifieur.pt")

            print("Models initialized successfully.")
        except NewConnectionError as conn_error:
            print(f"Connection to Ollama failed: {conn_error}")
            raise RuntimeError(
                "Failed to connect to Ollama. Check the service URL and availability."
            )
        except Exception as e:
            print(f"Error initializing models: {e}")
            raise RuntimeError("General error during model initialization.")

    #def test_ollama_embeddings(self):
        """
        Appelle la fonction globale pour vérifier la génération d'embeddings Ollama.
        """
        #test_ollama_embeddings(self)

    async def on_startup(self):
        """
        Exemple de chargement de documents et création d'un index.
        """
        print("Loading documents...")
        self.documents = SimpleDirectoryReader("./rag/").load_data()

        print("Building index with custom Ollama embeddings...")

        #self.embed_model = resolve_embed_model(self.embed_model)
        self.llm = Ollama(model='mistral-small:latest', request_timeout=30.0)
        #self.service_context = ServiceContext.from_defaults(embed_model=self.embed_model, llm=self.llm)
        #self.index = VectorStoreIndex.from_documents(self.documents, service_context = self.service_context)


        #self.index = VectorStoreIndex.from_documents(
        #    self.documents,
        #    embed_model=self.embed_model,
        #)
        print("Index built successfully.")


    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    async def inlet(self, body: dict, user: typing.Optional[dict] = None) -> dict:

        messages = body.get("messages", [])

        user_message = get_last_user_message(messages)
        
        return body

    async def outlet(self, body: dict, user: typing.Optional[dict] = None) -> dict:
        messages = body.get("messages", [])
        #assistant_message = get_last_assistant_message(messages)

        # Manipulez la réponse ici si nécessaire
        body["messages"] = messages
        return body
        """
        if assistant_message is not None:
            print("je passe dans le body")
            body = {**body, "messages": messages}
            return body

        
        # Manipulez la réponse ici si nécessaire
        body["messages"] = messages
        return body"""


    def classification(self, image):
        result = self.model_classif.predict(image, save=True, imgsz=640, conf=0.5)
        for r in result:
                if hasattr(r,'probs') and hasattr(r.probs,'top1'):
                    id=r.probs.top1
                    shipclass = self.model_classif.names[id]
                    #probs = result[0].probs  # get classification probabilities
                    #proba = probs.top1conf
                    #proba=float(proba.item())
                    proba = float(r.probs.top1conf.item())
        self.context_ship_class = shipclass
        return shipclass, proba

    def get_ship_class_info_llm(self, clasship, inputllm, img):
        try:
            model = OllamaLLM(
                model="mistral-small:latest", base_url="http://host.docker.internal:11434"
            )
            
            if img == True :
                response = model.generate(prompts=[f"que peux tu me dire sur les navire de type {clasship} ?"])
            elif clasship != None and inputllm != None :
                response = model.generate(prompts=[f"Au sujet des navires de type {clasship} répond à la question: {inputllm}"])
            else:
                response = model.generate(prompts=[f"{inputllm}"])

            # Vérification et extraction du texte
            if hasattr(response, "generations") and response.generations:
                reponse = response.generations[0][0].text.strip()
            else:
                reponse = "Pas de réponse générée."

        except Exception as e:
            reponse = f"Test failed: {e}"

        return reponse

    def get_ship_class_info_RAG(self, class_ship):
        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(f"que peux tu me dire sur les navire de type {class_ship} ?")

        return response

    def pipe(
        self,
        user_message: str,
        model_id: str = None,
        messages: typing.List[dict] = None,
        body: dict = None,
    ) -> typing.Union[str, typing.Generator, typing.Iterator]:

        clasship = None
        user_message = None
        ship_dict = {"burke": "Arleigh Burke", "PACDG": "porte avion charle de gaulle", "fremm":  "fremm", "PHA": "PHA mistral", "voilier": "voilier"}

        #obtenir l'encodage de l'image dans l'input du llm
        print(body)
        messages = body.get("messages", [])
        print("longueur de messages", len(messages))
        message_dict = messages[-1]
        messgelist = message_dict["content"]
        if type(messgelist[-1]) == dict:
            imageurl = messgelist[-1]["image_url"]  #dict: "type", "image_url"  POSSIBILITÉ DE RECUP DIRECTEMENT L IMAGE ?
            imgb64 = imageurl["url"]

        #by pass si l'utilisateur n'envoie pas d'image
        elif type(messgelist[-1]) == str :
            img = False
            messages = body.get("messages", [])
            user_message = get_last_user_message(messages)
            #rajouter oui ou non un context dans le prompt 
            if self.context_ship_class not None:
                clasship = ship_dict[self.context_ship_class]

            info =  self.get_ship_class_info_llm(clasship, user_message, img)
            return f'{info}'
        else :
            return f'le type est :{type(messgelist[-1])}'

        
        #decodage de l'image base64->png
        b = imgb64[imgb64.find("data"):imgb64.find(",")+1]
        print(b)
        imgb64 = imgb64.replace(b, "")
        for i in  range(imgb64.count("2025")):
            sentence = imgb64[imgb64.find("2025"):imgb64.find(" ")+1]
            imgb64 = imgb64.replace(sentence, "")
        imgb64 = imgb64.encode("ascii")
        with open("ship.png", "wb") as fh:
            fh.write(base64.decodebytes(imgb64))

        #prediction de l'image
        shipclass,prob = self.classification("ship.png")

        #obtenir des info depuis le llm
        print(ship_dict[shipclass])
        img = True
        info =  self.get_ship_class_info_llm(ship_dict[shipclass], user_message, img)

        try:
            return f"Classe de navire : {shipclass}\nProbabilité : {prob:.4f}\n\n{info}"
        except Exception as e:
            print(f"Error executing request: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


p = Pipeline()
print("test")


        