# import library
import streamlit as st
import pandas as pd
import json
import requests
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import folium
from streamlit_folium import folium_static

import tiktoken
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Mapbox Search API
MAPBOX_URL = "https://api.mapbox.com/geocoding/v5/mapbox.places/{query}.json"

def main():
    st.sidebar.title("Travel Recommendation App (Mapbox Edition)")

    mapbox_key = st.sidebar.text_input("Enter Mapbox API key:", type="password")
    openai_api_key = st.sidebar.text_input("Enter OpenAI API key:", type="password")

    st.sidebar.write('Please fill in the fields below.')
    destination = st.sidebar.text_input('Destination:', key='destination_app')
    radius = st.sidebar.number_input('Search Radius in meter:', value=3000, min_value=500, max_value=50000, step=100, key='radius_app')
    
    if destination and mapbox_key:
        # Get coordinates of destination
        def get_coordinates(place):
            url = MAPBOX_URL.format(query=place)
            params = {
                "access_token": mapbox_key,
                "limit": 1
            }
            res = requests.get(url, params=params)
            if res.status_code == 200:
                data = res.json()
                if data["features"]:
                    coords = data["features"][0]["geometry"]["coordinates"]
                    return coords[1], coords[0]  # lat, lon
            return None, None

        initial_latitude, initial_longitude = get_coordinates(destination)
        if not initial_latitude:
            st.error("Could not find destination. Try another input.")
            return

        circle_center = {"latitude": initial_latitude, "longitude": initial_longitude}
        circle_radius = radius

        # Mapbox place search (category simulation)
        def search_places(query, place_type):
            url = MAPBOX_URL.format(query=query)
            params = {
                "access_token": mapbox_key,
                "limit": 5,
                "proximity": f"{circle_center['longitude']},{circle_center['latitude']}"
            }
            res = requests.get(url, params=params)
            if res.status_code == 200:
                data = res.json()
                if data["features"]:
                    df = pd.DataFrame([{
                        "Name": f["text"],
                        "Address": f["place_name"],
                        "Latitude": f["geometry"]["coordinates"][1],
                        "Longitude": f["geometry"]["coordinates"][0],
                        "Type": place_type,
                        "Rating": None,   # Mapbox has no ratings
                        "User Rating Count": None,
                        "Website URL": None,
                        "Google Maps URL": None
                    } for f in data["features"]])
                    return df
            return pd.DataFrame()

        def hotel():
            return search_places(f"hotel near {destination}", "Hotel")

        def restaurant():
            return search_places(f"restaurant near {destination}", "Restaurant")

        def tourist():
            return search_places(f"tourist attraction near {destination}", "Tourist")

        df_hotel = hotel()
        df_restaurant = restaurant()
        df_tourist = tourist()

        df_place = pd.concat([df_hotel, df_restaurant, df_tourist], ignore_index=True)

        # Keep columns uniform
        df_place_rename = df_place[["Type","Name","Address","Rating","User Rating Count","Google Maps URL","Website URL","Latitude","Longitude"]]

        def database():
            st.dataframe(df_place_rename)

        def maps():
            st.header("🌏 Travel Recommendation App (Mapbox Edition) 🌏")

            places_type = st.radio('Looking for: ',["Hotels 🏨", "Restaurants 🍴","Tourist Attractions ⭐"])
            initial_location = [initial_latitude, initial_longitude]
            type_colour = {'Hotel':'blue', 'Restaurant':'green', 'Tourist':'orange'}
            type_icon = {'Hotel':'home', 'Restaurant':'cutlery', 'Tourist':'star'}

            if places_type == 'Hotels 🏨': 
                df_place = df_hotel
            elif places_type == 'Restaurants 🍴':
                df_place = df_restaurant
            else:
                df_place = df_tourist

            with st.spinner("Just a moment..."):
                mymap  = folium.Map(location = initial_location, zoom_start=12, control_scale=True)
                for index,row in df_place.iterrows():
                    location = [row['Latitude'], row['Longitude']]
                    content = (str(row['Name']) + '<br>' + 
                               'Address: ' + str(row['Address']))
                    iframe = folium.IFrame(content, width=250, height=100)
                    popup = folium.Popup(iframe, max_width=300)

                    icon_color = type_colour[row['Type']]
                    icon_type = type_icon[row['Type']]
                    icon = folium.Icon(color=icon_color, icon=icon_type)

                    folium.Marker(location=location, popup=popup, icon=icon).add_to(mymap)

                folium_static(mymap)

        def chatbot():
            class Message(BaseModel):
                actor: str
                payload : str

            llm = ChatOpenAI(openai_api_key=openai_api_key,model_name='gpt-3.5-turbo', temperature=0) 

            USER = "user"
            ASSISTANT = "ai"
            MESSAGES = "messages"

            if MESSAGES not in st.session_state:
                st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi! How can I help you?")]
            
            msg: Message
            for msg in st.session_state[MESSAGES]:
                st.chat_message(msg.actor).write(msg.payload)

            query: str = st.chat_input("Enter a prompt here")

            if not df_place.empty:
                df_place['combined_info'] = df_place.apply(lambda row: f"Type: {row['Type']}, Name: {row['Name']}. Address: {row['Address']}", axis=1)
                loader = DataFrameLoader(df_place, page_content_column="combined_info")
                docs  = loader.load()

                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(docs)

                modelPath = "sentence-transformers/all-MiniLM-l6-v2"
                model_kwargs = {'device':'cpu'}
                encode_kwargs = {'normalize_embeddings': False}

                embeddings = HuggingFaceEmbeddings(
                    model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
                )

                vectorstore  = FAISS.from_documents(texts, embeddings)

                template = """ 
                You are a travel assistant. From the following context and chat history, 
                assist users in finding recommendations. Provide three recommendations with address.
                
                {context}

                chat history: {history}

                input: {question} 
                Your Response:
                """

                prompt = PromptTemplate(
                    input_variables=["context","history","question"],
                    template=template,
                )

                memory = ConversationBufferMemory(memory_key="history", input_key="question", return_messages=True)
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type='stuff',
                    retriever=vectorstore.as_retriever(),
                    verbose=True,
                    chain_type_kwargs={
                        "verbose": True,
                        "prompt": prompt,
                        "memory": memory}
                )
                if query:
                    st.session_state[MESSAGES].append(Message(actor=USER, payload=str(query)))
                    st.chat_message(USER).write(query)

                    with st.spinner("Please wait..."):
                        response: str = qa.run(query = query)
                        st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
                        st.chat_message(ASSISTANT).write(response)
        method = st.sidebar.radio(" ",["Search 🔎","ChatBot 🤖","Database 📑"], key="method_app")
        if method == "Search 🔎":
            maps()
        elif method == "ChatBot 🤖":
            chatbot()
        else:
            database()
    st.sidebar.markdown(''' 
        ## Created by: 
        Adapted for Mapbox 🚀
        ''')
if __name__ == '__main__':

    main()
