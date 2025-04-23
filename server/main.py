from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import OpenAI 
import traceback
import numpy as np
import cv2
import base64
import requests
import json
import time
import random
import string
import threading
import re
from langgraph.graph import StateGraph
from typing import Dict, List
from pydantic import BaseModel
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import io
import os
import mailtrap as mt
from dotenv import load_dotenv
from mailtrap import Mail, Address, Attachment, Disposition, MailtrapClient
from twilio.rest import Client
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input, add, concatenate
from keras.models import Model

app = Flask(__name__)
CORS(app, origins=['https://projectreleaf.xyz'])
load_dotenv()
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["5 per minute"]
)

COST_PER_PREDICTION = 0.05 
COST_PER_TILE = 0.001      
COST_PER_TOKEN = 0.000002  
WATCHLIST_FILE = 'watchlist.json'
MAILTRAP_API_KEY = os.getenv("MAILTRAP_API_KEY")
SENDER_EMAIL = "releaf@demomailtrap.com"
SENDER_NAME = "ReLeaf Forest Monitor"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

account_sid = os.getenv("TWILIO_SID")
auth_token = os.getenv("TWILIO_TOKEN")
tw_client = Client(account_sid, auth_token)

num_classes = 7

input_layer = Input(shape=(512, 512, 3), name="unet_input")
encoder = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="en_block1_conv1")(input_layer)
encoder = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="en_block1_conv2")(encoder)
block1_output = encoder

encoder = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block1_mxp")(encoder)

encoder = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="en_block2_conv1")(encoder)
encoder = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="en_block2_conv2")(encoder)
block2_output = encoder

encoder = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block2_mxp")(encoder)

encoder = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="en_block3_conv1")(encoder)
encoder = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="en_block3_conv2")(encoder)
block3_output = encoder

encoder = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block3_mxp")(encoder)

encoder = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="en_block4_conv1")(encoder)
encoder = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="en_block4_conv2")(encoder)
block4_output = encoder

encoder_output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="en_block4_mxp")(encoder)
bottleneck = Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="btn_conv1")(encoder_output)
bottleneck = Conv2D(filters=1024, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="btn_conv2")(bottleneck)

bottleneck_output = UpSampling2D(size=(2, 2), data_format=None, interpolation="nearest", name="btn_output")(bottleneck)
decoder = Conv2D(filters=1024, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="de_block1_conv1")(bottleneck_output)
decoder = concatenate([block4_output, decoder])
decoder = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="de_block1_conv2")(decoder)

decoder = UpSampling2D(size=(2, 2), data_format=None, interpolation="nearest", name="de_block1_up")(decoder)

decoder = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="de_block2_conv1")(decoder)
decoder = concatenate([block3_output, decoder])
decoder = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="de_block2_conv2")(decoder)

decoder = UpSampling2D(size=(2, 2), data_format=None, interpolation="nearest", name="de_block2_up")(decoder)

decoder = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="de_block3_conv1")(decoder)
decoder = concatenate([block2_output, decoder])
decoder = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="de_block3_conv2")(decoder)

decoder = UpSampling2D(size=(2, 2), data_format=None, interpolation="nearest", name="de_block3_up")(decoder)

decoder = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="de_block4_conv1")(decoder)
decoder = concatenate([block1_output, decoder])
decoder = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="de_block4_conv2")(decoder)
decoder = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding='same', strides=(1, 1), name="de_block4_conv3")(decoder)

decoder_output = Conv2D(filters=num_classes, kernel_size=(1, 1), activation="softmax", padding='same', strides=(1, 1), name="final_output")(decoder)

model = Model(inputs=[input_layer], outputs=[decoder_output])
model.load_weights('best_model.keras')

client = OpenAI(api_key=OPENAI_API_KEY)

try:
    with open(WATCHLIST_FILE, 'r') as f:
        watchlist = json.load(f)
except FileNotFoundError:
    watchlist = {}

def save_watchlist():
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(watchlist, f, indent=4)

def get_next_report_time(timeframe):
    now = datetime.now()
    if timeframe == "Daily":
        return now + timedelta(days=1)
    elif timeframe == "Weekly":
        return now + timedelta(weeks=1)
    else:
        return now + timedelta(days=30)

def generate_pie_chart(forested, deforested, other):
    plt.figure(figsize=(8, 8))
    plt.pie([forested, deforested, other],
            labels=['Forested', 'Deforested', 'Other'],
            colors=['green', 'red', 'blue'],
            autopct='%1.1f%%')
    plt.title('Land Composition Analysis')

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    return buffer.read()

def get_ai_analysis(current_data, location, previous_data=None):
    class GeoQueryState(BaseModel):
        query: str
        subqueries: List[str] = []
        results: Dict[str, str] = {}
        final_summary: str = ""
        analysis_steps: List[Dict[str, str]] = []

    geo_graph = StateGraph(GeoQueryState)
    
    def extract_location(query: str) -> str:
        match = re.search(r"(?:near|in)\s(.+)", query, re.IGNORECASE)
        return match.group(1).strip() if match else "Unknown Location"

    def break_query(state: GeoQueryState) -> GeoQueryState:
        location = extract_location(state.query)
        state.subqueries = [
            f"Deforestation activity in {location}",
            f"Latest reports on deforestation in {location}",
            f"Environmental impact studies in {location}"
        ]
        
        state.analysis_steps.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": "Breaking Down Query",
            "subtitle": "Identified key data points to fetch",
            "description": f"Generated subqueries: {', '.join(state.subqueries)}"
        })
        return state

    def fetch_web_data(state: GeoQueryState) -> GeoQueryState:
        for subquery in state.subqueries:
            url = "https://google.serper.dev/search"
            payload = {"q": subquery}
            headers = {
                "X-API-KEY": os.getenv("SERPER_API_KEY"),
                "Content-Type": "application/json",
            }
            response = requests.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                state.results[subquery] = response.text[:2000]
            else:
                state.results[subquery] = "Error fetching data."
        
        state.analysis_steps.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": "Fetching Web Data",
            "subtitle": "Retrieved real-time data on deforestation",
            "description": "Successfully fetched data for deforestation queries. Errors handled where necessary."
        })
        return state
    
    def summarize_and_infer(state: GeoQueryState) -> GeoQueryState:
        combined_data = "\n\n".join(
            [f"{query}:\n{content}" for query, content in state.results.items()]
        )
        if previous_data:
            prompt = f"""
            Analyze the following geospatial data and focus on deforestation trends. Provide key insights concisely.

            🚨 Anomaly Detected? (✅ Yes / ❌ No - Only if there is a clear spike in deforestation specific to this area)
            📌 Key Observations: (One-liner summary of findings)
            📊 Deforestation Breakdown:
            - 🌲 Deforestation: (value) (🟢 Normal / 🔴 Anomaly / Not Available)
            📉 Deviation from Normal: (If any)
            🔍 Final Conclusion: (Short insight)

            Data:
            {combined_data}
            Data from Semantic Segmentation Model:
            {current_data}
            Data from Previous Report:
            {previous_data}
            Ensure that an anomaly is only marked if a strong connection to the provided data exists.
            """
        else:
            prompt = f"""
            Analyze the following geospatial data and focus on deforestation trends. Provide key insights concisely.

            🚨 Anomaly Detected? (✅ Yes / ❌ No - Only if there is a clear spike in deforestation specific to this area)
            📌 Key Observations: (One-liner summary of findings)
            📊 Deforestation Breakdown:
            - 🌲 Deforestation: (value) (🟢 Normal / 🔴 Anomaly / Not Available)
            📉 Deviation from Normal: (If any)
            🔍 Final Conclusion: (Short insight)

            Data:
            {combined_data}
            Data from Semantic Segmentation Model:
            {current_data}
            Ensure that an anomaly is only marked if a strong connection to the provided data exists.
            """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are an expert in environmental data analysis."},
                        {"role": "user", "content": prompt}]
            )
            state.final_summary = response.choices[0].message.content if response.choices else "No insight available."
        except Exception as e:
            state.final_summary = f"Error generating response: {str(e)}"
        
        state.analysis_steps.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": "Summarizing & Analyzing",
            "subtitle": "Extracting key insights on deforestation",
            "description": state.final_summary
        })
        return state

    def take_action(state: GeoQueryState) -> GeoQueryState:
          location = extract_location(state.query)
          community_query = f"environmental organizations near {location}"
          contact_query = f"contact details of {location}"
          
          def search(query):
              response = requests.post(
                  "https://google.serper.dev/search",
                  headers={"X-API-KEY": os.getenv("SERPER_API_KEY"), "Content-Type": "application/json"},
                  json={"q": query}
              )
              return response.text[:2000] if response.status_code == 200 else "Error fetching data."
          
          community_results = search(community_query)
          contact_results = search(contact_query)
          
          prompt = f"""
            You are an expert information extractor. From the given search results, extract **only the most relevant and valid contact details** (phone numbers and emails) of **environmental or community-related organizations**.

            Return your response in this structured format:
            1. **Organization Name**
            - 📍Location Type: (Company, NGO, School, Government Office, etc.)
            - 🤝Community Focus: (One-line summary of their environmental/community work)
            - 📞Contact Details: (Only phone numbers and/or emails that clearly belong to the organization)

            ⚠️ Do NOT include irrelevant businesses, advertisements, or repeated entries.
            ⚠️ Only include entries with **clear and trustworthy** contact details.

            -- COMMUNITY RESULTS --
            {community_results}

            -- CONTACT RESULTS --
            {contact_results}
            """
          try:
              response = client.chat.completions.create(
                  model="gpt-4o",
                  messages=[{"role": "system", "content": "Extract structured details."},
                            {"role": "user", "content": prompt}]
              )
              llm_output = response.choices[0].message.content
              llm_output = llm_output.replace("*","")
          except Exception as e:
              llm_output = f"Error processing data: {str(e)}"
          state.analysis_steps.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": "Searching for Contacts",
            "subtitle": "Looking for local communities",
            "description": llm_output
            })
          state.results["Actionable Summary"] = llm_output
          
          anomaly_detected = "✅ Yes" in state.final_summary
          contact_found = "Phone:" in llm_output or "Email:" in llm_output
          
          if anomaly_detected and contact_found:
              email_body = f"""
              Subject: Urgent Deforestation Alert Near {location}
              
              Dear Concerned Authority,
              
              Our latest environmental analysis has detected unusual deforestation activity in {location}. Here are the key findings:
              
              {state.final_summary}
              
              We believe immediate action is necessary. Please let us know how we can collaborate.
              
              Best Regards,
              ReLeaf Team
              """
              state.results["Take Action"] = "Yes"
              state.results["Draft Email"] = email_body
              state.analysis_steps.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "title": "Taking Action",
                "subtitle": "Sending email",
                "description": email_body
                })
              def extract_email(text):
                match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
                return match.group(0) if match else None
              email = extract_email(llm_output)
              mt_client = MailtrapClient(token=MAILTRAP_API_KEY)
              mail = Mail(
                sender=Address(
                    email=SENDER_EMAIL,
                    name=SENDER_NAME
                ),
                to=[Address(email=email)],
                subject=f'Urgent Deforestation Alert Near {location}',
                html=email_body,
              )
              mt_client.send(mail)
          else:
              state.results["Take Action"] = "No"
          
          return state
    
    geo_graph.add_node("break_query", break_query)
    geo_graph.add_node("fetch_web_data", fetch_web_data)
    geo_graph.add_node("summarize_and_infer", summarize_and_infer)
    geo_graph.add_node("take_action", take_action)
    
    geo_graph.add_edge("break_query", "fetch_web_data")
    geo_graph.add_edge("fetch_web_data", "summarize_and_infer")
    geo_graph.add_edge("summarize_and_infer","take_action")
    geo_graph.set_entry_point("break_query")
    geo_graph.set_finish_point(["take_action"])
    
    geo_compiled = geo_graph.compile()
    
    state = GeoQueryState(query=f"Analyze deforestation trends and take action near {location}")
    output = geo_compiled.invoke(state)
    def generate_html(analysis_steps):
        html_template = """
            <div class="email-container">
                <div class="title">ReLeaf AI Agent Decision Flow</div>
                <div class="subtitle">How the AI processed this request</div>
        """
        
        html_template += """
                <div class="step bg-gray-900">
                    <strong>🟢 AI Agent Activated</strong>
                    <p>The AI Agent has been activated and is ready to process the query.</p>
                </div>
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                    <tr>
                        <td align="center">
                            <img src="https://api.projectreleaf.xyz/static/arrow-down-solid.png" width="20" style="display: block;">
                        </td>
                    </tr>
                </table>
        """
        
        color_mapping = {
            "Breaking Down Query": "bg-green-700",
            "Fetching Web Data": "bg-blue-700",
            "Summarizing & Analyzing": "bg-yellow-700",
            "Searching for Contacts": "bg-green-700",
            "Taking Action": "bg-red-700"
        }
        
        for i, step in enumerate(analysis_steps):
            color_class = color_mapping.get(step['title'], "bg-gray-900")
            description_html = step['description'].replace('\n', '<br/>')
            html_template += f"""
                <div class="step {color_class}">
                    <strong>{step['title']}</strong>
                    <p>{description_html}</p>
                </div>
            """
            if i < len(analysis_steps) - 1:
                html_template += """
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                    <tr>
                        <td align="center">
                            <img src="https://api.projectreleaf.xyz/static/arrow-down-solid.png" width="20" style="display: block;">
                        </td>
                    </tr>
                </table>
                """
        
        html_template += """
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                    <tr>
                        <td align="center">
                            <img src="https://api.projectreleaf.xyz/static/arrow-down-solid.png" width="20" style="display: block;">
                        </td>
                    </tr>
                </table>
                <div class="step bg-gray-900">
                    <strong>✅ AI Agent Process Completed</strong>
                    <p>The AI Agent has finished processing the request.</p>
                </div>
            </div>
        """
        
        return html_template


    steps = output['analysis_steps']
    html_output = generate_html(steps)
    return output['final_summary'], html_output

def calculate_costs(tile_count):
    prediction_cost = COST_PER_PREDICTION
    tile_costs = tile_count * COST_PER_TILE
    ai_analysis_cost = 0.02  

    total_cost = prediction_cost + tile_costs + ai_analysis_cost
    return {
        "prediction_cost": prediction_cost,
        "tile_costs": tile_costs,
        "ai_analysis_cost": ai_analysis_cost,
        "total_cost": total_cost
    }

def generate_comparison_image(original_image, masked_image,map=True):
    target_size = (512, 512)
    original_resized = cv2.resize(original_image, target_size)
    masked_resized = cv2.resize(masked_image, target_size)

    plt.figure(figsize=(12, 8), facecolor='white')
    
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    plt.subplot(121)
    if map:
        plt.imshow(original_resized)
    else:
        plt.imshow(cv2.cvtColor(original_resized,cv2.COLOR_BGR2RGB))
    plt.title('Satellite Image')
    plt.axis('off')
    plt.gca().set_facecolor('white')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(masked_resized, cv2.COLOR_BGR2RGB))
    plt.title('Forest Analysis')
    plt.axis('off')
    plt.gca().set_facecolor('white')

    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#3C1098', label='Building'),      
        plt.Rectangle((0,0),1,1, facecolor='#8429F6', label='Land'),              
        plt.Rectangle((0,0),1,1, facecolor='#FEDD3A', label='Vegetation'),    
        plt.Rectangle((0,0),1,1, facecolor='#E2A929', label='Water'),         
        plt.Rectangle((0,0),1,1, facecolor='#E8623C', label='Object'),        
        plt.Rectangle((0,0),1,1, facecolor='#000000', label='Unlabeled')  
    ]
    
    plt.figlegend(handles=legend_elements, loc='lower center', ncol=4, 
                 bbox_to_anchor=(0.5, 0.0), facecolor='white')

    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', 
                bbox_inches='tight', 
                facecolor='white', 
                edgecolor='none', 
                pad_inches=0.1,
                dpi=300)
    plt.close()
    buffer.seek(0)
    return buffer.read()

def get_location_name(lat, lon):
    try:
        headers = {'User-Agent': 'ReLeaf Forest Monitor/1.0'}
        response = requests.get(
            f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}",
            headers=headers
        )
        if response.status_code == 200:
            data = response.json()
            address = data.get('address', {})

            location_parts = []
            for key in ['house_number', 'road', 'neighbourhood', 'suburb', 'city', 'county', 'state', 'country']:
                if address.get(key):
                    location_parts.append(address[key])

            if location_parts:
                return ", ".join(location_parts)

    except Exception as e:
        print(f"Error getting location name: {e}")

    return "Unknown Address"

def create_html_report(region_data, current_results, previous_results, comparison_image, 
                      pie_chart, ai_analysis, decision_flow, costs):
    current_time = datetime.now()
    formatted_date = current_time.strftime('%B %d, %Y')
    formatted_time = current_time.strftime('%I:%M %p')
    BASE62 = string.digits + string.ascii_letters
    def encode_base62(num, length=8):
        result = ''
        while num > 0:
            result = BASE62[num % 62] + result
            num //= 62
        return result.rjust(length, random.choice(BASE62))  # Pad if too short
    
    def generate_unique_code(length=10):
        timestamp = int(time.time() * 1e6)  # microseconds
        random_bits = random.randint(0, 9999)
        combined = (timestamp << 14) | random_bits  # shift to combine both
        return encode_base62(combined, length)[:length]
    unique = generate_unique_code()
    if 'location' in region_data:
        location = region_data['location']
    else:
        location = get_location_name(
            region_data['min_lat'], 
            region_data['min_long']
        )
    
    if previous_results:
        has_changes = (
            abs(current_results["forested_percentage"] - previous_results["forested_percentage"]) > 0.1 or
            abs(current_results["deforested_percentage"] - previous_results["deforested_percentage"]) > 0.1 or
            abs(current_results["other_percentage"] - previous_results["other_percentage"]) > 0.1
        )

        if has_changes:
            comparison_section = f"""
                <div class="section">
                    <h2>Historical Comparison</h2>
                    <div class="comparison-grid">
                        <div class="comparison-card current">
                            <h3>Current Analysis</h3>
                            <p>Forest Coverage: <span class="highlight">{current_results["forested_percentage"]:.1f}%</span></p>
                            <p>Non-forest Areas: <span class="highlight">{current_results["deforested_percentage"]:.1f}%</span></p>
                            <p>Other: <span class="highlight">{current_results["other_percentage"]:.1f}%</span></p>
                        </div>
                        <div class="comparison-card previous">
                            <h3>Previous Analysis</h3>
                            <p>Forest Coverage: <span class="highlight">{previous_results["forested_percentage"]:.1f}%</span></p>
                            <p>Non-forest Areas: <span class="highlight">{previous_results["deforested_percentage"]:.1f}%</span></p>
                            <p>Other: <span class="highlight">{previous_results["other_percentage"]:.1f}%</span></p>
                        </div>
                    </div>
                </div>
            """
        else:
            comparison_section = f"""
                <div class="section">
                    <h2>Historical Comparison</h2>
                    <div class="comparison-card">
                        <p>N/A - No significant changes detected since the last report.</p>
                        <p>Current composition remains at:</p>
                        <ul>
                            <li>Forest Coverage: <span class="highlight">{current_results["forested_percentage"]:.1f}%</span></li>
                            <li>Non-forest Areas: <span class="highlight">{current_results["deforested_percentage"]:.1f}%</span></li>
                            <li>Other: <span class="highlight">{current_results["other_percentage"]:.1f}%</span></li>
                        </ul>
                    </div>
                </div>
            """
    else:
        comparison_section = f"""
            <div class="section">
                <h2>Historical Comparison</h2>
                <div class="comparison-card first-report">
                    <p>N/A - This is your first report for this region.</p>
                    <p>Current composition:</p>
                    <ul>
                        <li>Forest Coverage: <span class="highlight">{current_results["forested_percentage"]:.1f}%</span></li>
                        <li>Non-forest Areas: <span class="highlight">{current_results["deforested_percentage"]:.1f}%</span></li>
                        <li>Other: <span class="highlight">{current_results["other_percentage"]:.1f}%</span></li>
                    </ul>
                </div>
            </div>
        """

    html_template = f"""
    <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
            <title>ReLeaf Forest Coverage Report</title>
            <link rel="icon" type="image/x-icon" href="favicon.ico">
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #1e4d92, #2c7744);
                    color: white;
                    padding: 30px;
                    text-align: center;
                    border-radius: 10px 10px 0 0;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 28px;
                }}
                .header p {{
                    margin: 10px 0 0;
                    opacity: 0.9;
                }}
                .section {{
                    background: white;
                    padding: 25px;
                    margin-bottom: 25px;
                    border-radius: 10px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .section h2 {{
                    color: #1e4d92;
                    margin-top: 0;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 10px;
                }}
                .image-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .image-container img {{
                    max-width: 100%;
                    border-radius: 8px;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                }}
                .comparison-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-top: 20px;
                }}
                .comparison-card {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    border: 1px solid #e9ecef;
                }}
                .comparison-card.current {{
                    border-left: 4px solid #2c7744;
                }}
                .comparison-card.previous {{
                    border-left: 4px solid #1e4d92;
                }}
                .comparison-card.first-report {{
                    grid-column: 1 / -1;
                    text-align: center;
                    background: #e8f4ea;
                }}
                .highlight {{
                    font-weight: bold;
                    color: #2c7744;
                }}
                .costs {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin-top: 20px;
                }}
                .costs h3 {{
                    color: #1e4d92;
                    margin-top: 0;
                }}
                .location-info {{
                    background: #e8f4ea;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    font-size: 1.1em;
                }}
                ul {{
                    list-style-type: none;
                    padding-left: 0;
                }}
                ul li {{
                    margin-bottom: 10px;
                }}
                .email-container {{ background-color: #1e1e1e; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.5); max-width: 600px; margin: auto; text-align: center; }}
                .title {{ font-size: 26px; font-weight: bold; color: #ffffff; }}
                .subtitle {{ font-size: 18px; color: #b0b0b0; margin-bottom: 20px; }}
                .step {{ padding: 15px; border-radius: 10px; margin: 10px 0; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.5); color: #ffffff; text-align: left; position: relative; }}
                .step:hover {{ box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.7); transform: translateY(-2px); transition: all 0.3s ease; }}
                .bg-gray-900 {{ background-color: #2c3e50; }}
                .bg-green-700 {{ background-color: #27ae60; }}
                .bg-blue-700 {{ background-color: #2980b9; }}
                .bg-yellow-700 {{ background-color: #f39c12; }}
                .bg-red-700 {{ background-color: #c0392b; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>ReLeaf Forest Coverage Report</h1>
                    <p>{formatted_date} at {formatted_time} UTC</p>
                </div>

                <div class="section">
                    <h2>Area Analysis</h2>
                    <div class="location-info">
                        <p>Monitoring region: {location}</p>
                    </div>
                    <div class="image-container">
                        <img src="cid:comparison_image.png" alt="Forest Analysis Comparison">
                    </div>
                </div>

                <div class="section">
                    <h2>Composition Analysis</h2>
                    <div class="image-container">
                        <img src="cid:pie_chart.png" alt="Land Composition">
                    </div>
                </div>

                {comparison_section}

                <div class="section">
                    <h2>AI Analysis</h2>
                    <p>{ai_analysis.replace(chr(10), '<br>')}</p>
                </div>
                <div class="section decision-flow">
                    <h2>Decision Flow Analysis</h2>
                    {decision_flow}
                </div>
                <div class="section">
                    <h2>Report Details</h2>
                    <div class="costs">
                        <h3>Processing Information</h3>
                        <p>Prediction Processing: ${costs['prediction_cost']:.3f}</p>
                        <p>Satellite Imagery: ${costs['tile_costs']:.3f}</p>
                        <p>AI Analysis: ${costs['ai_analysis_cost']:.3f}</p>
                        <p><strong>Total Cost: ${costs['total_cost']:.3f}</strong></p>
                        <div class="section">
                            <h2>Shareable Link</h2>
                            <p>You can access this report later using the following unique link:</p>
                            <p>
                                <a href="https://api.projectreleaf.xyz/static/{unique}.html" target="_blank">
                                    https://api.projectreleaf.xyz/static/{unique}.html
                                </a>
                            </p>
                        </div>
                </div>
            </div>
        </body>
    </html>
    """
    return unique, html_template

def send_email_report(email, region_data, prediction_results, comparison_image, number, map=True):
    print("in function")
    try:
        previous_results = None
        if map:
            if email in watchlist and region_data.get('previous_results'):
                previous_results = region_data['previous_results']
                subject = f"ReLeaf Forest Coverage Report - {region_data['timeframe']}"
            else:
                subject = f"ReLeaf Forest Coverage Report - One Time"
            tile_count = ((region_data['max_lat'] - region_data['min_lat']) * 
                 (region_data['max_long'] - region_data['min_long']) * 
                 (2 ** region_data['zoom']))
            location = get_location_name(
            region_data['min_lat'], 
            region_data['min_long']
        )
        else:
            if 'location' in region_data:
                location = region_data['location']
            subject = "ReLeaf Forest Coverage Report - Custom (One Time)"
            tile_count = 4

        pie_chart_image = generate_pie_chart(
            prediction_results['forested_percentage'],
            prediction_results['deforested_percentage'],
            prediction_results['other_percentage']
        )
        ai_analysis, decision_flow = get_ai_analysis(prediction_results, location, previous_results)
        prediction_results['ai_analysis'] = ai_analysis.strip("*")
        costs = calculate_costs(tile_count)

        unique, html_content = create_html_report(
            region_data,
            prediction_results,
            previous_results,
            "comparison_image.png",
            "pie_chart.png",
            ai_analysis,
            decision_flow,
            costs
        )

        mail = Mail(
            sender=Address(
                email=SENDER_EMAIL,
                name=SENDER_NAME
            ),
            to=[Address(email=email)],
            subject=subject,
            html=html_content,
            attachments=[
                Attachment(
                    content=base64.b64encode(comparison_image),
                    filename="comparison_image.png",
                    disposition=Disposition.INLINE,
                    mimetype="image/png",
                    content_id="comparison_image.png"
                ),
                Attachment(
                    content=base64.b64encode(pie_chart_image),
                    filename="pie_chart.png",
                    disposition=Disposition.INLINE,
                    mimetype="image/png",
                    content_id="pie_chart.png"
                )
            ]
        )
        html_content = re.sub(
    r'<div class="section">\s*<h2>Shareable Link</h2>.*?</div>',
    '',
    html_content,
    flags=re.DOTALL
)
        def embed_images_into_html(html, image_dict):
            for cid, img_bytes in image_dict.items():
                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                html = html.replace(f"cid:{cid}", f"data:image/png;base64,{b64_img}")
            return html
            
        final_html = embed_images_into_html(
            html_content,
            {
                "comparison_image.png": comparison_image,
                "pie_chart.png": pie_chart_image
            }
        )
        with open(f"static/{unique}.html", "w", encoding="utf-8") as f: f.write(final_html)
        mt_client = MailtrapClient(token=MAILTRAP_API_KEY)
        mt_client.send(mail)
        if number != "":
            image = Image.open(BytesIO(comparison_image))
            path = f"static/{unique}.png"
            image.save(path, format="PNG")
            message = tw_client.messages.create(
                body=f"Here's a link to view your ReLeaf forest coverage report: https://api.projectreleaf.xyz/static/{unique}.html",
                media_url=[f"https://api.projectreleaf.xyz/static/{unique}.png"],
                to=f"whatsapp:+91{number}",
                from_="whatsapp:+14155238886",
            )
            print(message.sid)
        print(f"Successfully sent report to {email}")
        return comparison_image

    except Exception as e:
        print(f"Failed to send email to {email}: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

def process_image(image):
    image_preprocessed = preprocess_image(image)
    prediction = model.predict(image_preprocessed)[0]
    print("Predicted")
    output_image = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    print("Created an output image")
    class_mapping = {
        0: 'building',
        1: 'land',
        2: 'road',
        3: 'vegetation',
        4: 'water',
        5: 'object',
        6: 'unlabeled'
    }
    
    color_map = {
        0: (152, 16, 60),
        1: (246, 41, 132),
        2: (246, 41, 132),
        3: (58, 221, 254),
        4: (41, 169, 226),
        5: (60, 98, 232),
        6: (0, 0, 0)
    }
    
    class_indices = np.argmax(prediction, axis=-1)
    print("Class distribution:")
    for idx in range(7):
        mask = (class_indices == idx)
        percentage = np.sum(mask) / class_indices.size * 100
        print(f"Class {idx} ({class_mapping[idx]}): {percentage:.2f}%")
    
    for idx in range(7):
        mask = (class_indices == idx)
        output_image[mask] = color_map[idx]
    
    total_pixels = prediction.shape[0] * prediction.shape[1]
    
    forested_mask = (class_indices == 3)
    forested_percentage = np.sum(forested_mask) / total_pixels * 100
    
    deforested_mask = (class_indices == 1)
    deforested_percentage = np.sum(deforested_mask) / total_pixels * 100
    
    other_mask = (class_indices == 0) | (class_indices == 2) | (class_indices == 4) | (class_indices == 5) | (class_indices == 6)
    other_percentage = np.sum(other_mask) / total_pixels * 100
    
    print(f"\nGrouped percentages:")
    print(f"Forested (Vegetation): {forested_percentage:.2f}%")
    print(f"Deforested (Building + Land): {deforested_percentage:.2f}%")
    print(f"Other (Road + Water + Object + Unlabeled): {other_percentage:.2f}%")
    
    _, buffer = cv2.imencode('.png', output_image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "predicted_mask_base64": base64_image,
        "forested_percentage": float(forested_percentage),
        "deforested_percentage": float(deforested_percentage),
        "other_percentage": float(other_percentage)
    }

def preprocess_image(image):
    image_resized = cv2.resize(image, (512, 512))
    image_normalized = image_resized / 255.0
    return np.expand_dims(image_normalized, axis=0)

def lat_lon_to_tile(lat, lon, zoom):
    n = 2.0 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2.0 * n)
    return x_tile, y_tile

def fetch_and_stitch_tiles(min_lat, max_lat, min_lon, max_lon, zoom):
    tile_size = 256

    x_min, y_max = lat_lon_to_tile(max_lat, min_lon, zoom)
    x_max, y_min = lat_lon_to_tile(min_lat, max_lon, zoom)

    y_min, y_max = min(y_min, y_max), max(y_min, y_max)

    stitched_width = (x_max - x_min + 1) * tile_size
    stitched_height = (y_max - y_min + 1) * tile_size

    stitched_image = Image.new("RGB", (stitched_width, stitched_height))

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                tile = Image.open(io.BytesIO(response.content))
                stitched_image.paste(
                    tile, ((x - x_min) * tile_size, (y - y_min) * tile_size)
                )

    return np.array(stitched_image)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        email = data['email']
        results = None
        if 'image' in data:
            image = cv2.imdecode(
            np.frombuffer(
                base64.b64decode(data['image']),
                np.uint8
            ),
            cv2.IMREAD_COLOR
        )
            locaton = data['location']
            custom = True
        else:
            custom = False
            image = fetch_and_stitch_tiles(
                data['min_lat'],
                data['max_lat'],
                data['min_long'],
                data['max_long'],
                data['zoom']
            )

        results = process_image(image)
        masked_image = cv2.imdecode(
            np.frombuffer(
                base64.b64decode(results['predicted_mask_base64']),
                np.uint8
            ),
            cv2.IMREAD_COLOR
        )
        number = data['number']
        comparison_image = generate_comparison_image(image, masked_image,map=not custom)

        if data.get('watchlist'):
            if email not in watchlist:
                watchlist[email] = {}

            region_id = f"region_{len(watchlist[email])}"

            watchlist[email][region_id] = {
                'min_lat': data['min_lat'],
                'max_lat': data['max_lat'],
                'min_long': data['min_long'],
                'max_long': data['max_long'],
                'zoom': data['zoom'],
                'number': data['number'],
                'timeframe': data['timeframe'],
                'next_report': get_next_report_time(data['timeframe']).isoformat(),
                'previous_results': {
                    'forested_percentage': results['forested_percentage'],
                    'deforested_percentage': results['deforested_percentage'],
                    'other_percentage': results['other_percentage']
                }
            }

            send_email_report(email, watchlist[email][region_id], results, comparison_image,number)
            results['predicted_mask_base64'] = base64.b64encode(comparison_image).decode('utf-8')

            save_watchlist()
        else:
            if custom:
                print("sending report")
                send_email_report(email, data, results, comparison_image, number, map=False)
            else:
                region_data = {
                    'min_lat': data['min_lat'],
                    'max_lat': data['max_lat'],
                    'min_long': data['min_long'],
                    'max_long': data['max_long'],
                    'zoom': data['zoom'],
                    'previous_results': ''
                }
                send_email_report(email, region_data, results, comparison_image, number)
            if comparison_image is None:
                raise ValueError("Comparison image is None.")
    
            results['predicted_mask_base64'] = base64.b64encode(comparison_image).decode('utf-8')
        return jsonify(results)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

def check_and_send_reports():
    while True:
        current_time = datetime.now()

        for email, regions in list(watchlist.items()):
            for region_id, region_data in list(regions.items()):
                next_report = datetime.fromisoformat(region_data['next_report'])
                number = region_data['number']
                if current_time >= next_report:
                    image = fetch_and_stitch_tiles(
                        region_data['min_lat'],
                        region_data['max_lat'],
                        region_data['min_long'],
                        region_data['max_long'],
                        region_data['zoom']
                    )

                    results = process_image(image)
                    masked_image = cv2.imdecode(
                        np.frombuffer(
                            base64.b64decode(results['predicted_mask_base64']),
                            np.uint8
                        ),
                        cv2.IMREAD_COLOR
                    )
                    comparison_image = generate_comparison_image(image, masked_image,map=False)
                    previous_results = region_data.get('previous_results')
                    watchlist[email][region_id]['previous_results'] = {
                        'forested_percentage': results['forested_percentage'],
                        'deforested_percentage': results['deforested_percentage'],
                        'other_percentage': results['other_percentage']
                    }
                    print("Sending report")
                    send_email_report(email, region_data, results, comparison_image,number)

                    watchlist[email][region_id]['next_report'] = get_next_report_time(
                        region_data['timeframe']
                    ).isoformat()

                    save_watchlist()

        time.sleep(3600)

@app.errorhandler(429)
def ratelimit_error(e):
    return jsonify({"error": "Too many requests. Try again later."}), 429
    
def create_contact_confirmation_email(name):
    html_template = f"""
    <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #1e4d92, #2c7744);
                    color: white;
                    padding: 30px;
                    text-align: center;
                    border-radius: 10px 10px 0 0;
                }}
                .content {{
                    background: white;
                    padding: 30px;
                    border-radius: 0 0 10px 10px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .highlight {{
                    color: #2c7744;
                    font-weight: bold;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 20px;
                    color: #666;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Thank You for Contacting ReLeaf</h1>
                </div>
                <div class="content">
                    <p>Dear <span class="highlight">{name}</span>,</p>
                    <p>Thank you for reaching out to us regarding your inquiry. We have received your message and appreciate your interest in ReLeaf.</p>
                    <p>Our team will review your request and get back to you as soon as possible, typically within 24-48 hours.</p>
                    <p>In the meantime, feel free to explore our website and learn more about our mission to protect and monitor forest coverage worldwide.</p>
                    <p>Best regards,<br>The ReLeaf Team</p>
                </div>
                <div class="footer">
                    <p>© 2025 ReLeaf. All rights reserved.</p>
                </div>
            </div>
        </body>
    </html>
    """
    return html_template

def create_contact_notification_email(contact_data):
    html_template = f"""
    <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #1e4d92, #2c7744);
                    color: white;
                    padding: 30px;
                    text-align: center;
                    border-radius: 10px 10px 0 0;
                }}
                .content {{
                    background: white;
                    padding: 30px;
                    border-radius: 0 0 10px 10px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .contact-details {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .detail-row {{
                    display: flex;
                    margin-bottom: 10px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }}
                .detail-label {{
                    font-weight: bold;
                    width: 120px;
                    color: #1e4d92;
                }}
                .message-box {{
                    background: #e8f4ea;
                    padding: 20px;
                    border-radius: 8px;
                    margin-top: 20px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 20px;
                    color: #666;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>New Contact Form Submission</h1>
                </div>
                <div class="content">
                    <p>A new contact form has been submitted with the following details:</p>
                    
                    <div class="contact-details">
                        <div class="detail-row">
                            <div class="detail-label">Name:</div>
                            <div>{contact_data['name']}</div>
                        </div>
                        <div class="detail-row">
                            <div class="detail-label">Email:</div>
                            <div>{contact_data['email']}</div>
                        </div>
                        <div class="detail-row">
                            <div class="detail-label">Phone:</div>
                            <div>{contact_data['phone']}</div>
                        </div>
                        <div class="detail-row">
                            <div class="detail-label">Subject:</div>
                            <div>{contact_data['subject']}</div>
                        </div>
                    </div>

                    <div class="message-box">
                        <h3>Message:</h3>
                        <p>{contact_data['message']}</p>
                    </div>
                </div>
                <div class="footer">
                    <p>This is an automated notification from the ReLeaf contact system.</p>
                </div>
            </div>
        </body>
    </html>
    """
    return html_template

@app.route('/contact', methods=['POST'])
@limiter.limit("1 per day")
def handle_contact():
    try:
        data = request.get_json()
        
        subject_line = f"New Contact: {data['subject']} - ReLeaf"
        
        confirmation_mail = Mail(
            sender=Address(email=SENDER_EMAIL, name=SENDER_NAME),
            to=[Address(email=data['email'])],
            subject="Thank you for contacting ReLeaf",
            html=create_contact_confirmation_email(data['name'])
        )
        
        notification_mail = Mail(
            sender=Address(email=SENDER_EMAIL, name=SENDER_NAME),
            to=[
                Address(email="ved.shivane@gmail.com"),
                Address(email="bhaskarankeerthan@gmail.com"),
                Address(email="vjk5100@gmail.com")
            ],
            subject=subject_line,
            html=create_contact_notification_email(data)
        )
        
        mt_client = MailtrapClient(token=MAILTRAP_API_KEY)
        mt_client.send(confirmation_mail)
        mt_client.send(notification_mail)
        
        return jsonify({"success": True, "message": "Contact form submitted successfully"})
        
    except Exception as e:
        print(f"Error processing contact form: {str(e)}")
        return jsonify({"error": "Failed to process contact form"}), 500

if __name__ == "__main__":
    reporting_thread = threading.Thread(target=check_and_send_reports, daemon=True)
    reporting_thread.start()

    app.run(debug=False, host="0.0.0.0", port=80, use_debugger=False)
