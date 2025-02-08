import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, SeparableConv2D, BatchNormalization, 
                                     GroupNormalization, LeakyReLU, MaxPooling2D, Dropout, 
                                     Conv2DTranspose, concatenate, Activation)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras import applications, optimizers
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import model_to_dot, plot_model
from tensorflow.keras.callbacks import Callback

import numpy as np
import cv2
import base64
import requests
import json
import time
import threading
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import io
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from openai import OpenAI

import mailtrap as mt
from mailtrap import Mail, Address, Attachment, Disposition, MailtrapClient

app = Flask(__name__)
load_dotenv()
# Constants
WATCHLIST_FILE = 'watchlist.json'
MAILTRAP_API_KEY = os.getenv('MAILTRAP_API_KEY')
SENDER_EMAIL = "releaf@demomailtrap.com"
SENDER_NAME = "ReLeaf Forest Monitor"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Cost constants (example values - adjust as needed)
COST_PER_PREDICTION = 0.05  # Cost in USD per prediction
COST_PER_TILE = 0.001      # Cost per map tile
COST_PER_TOKEN = 0.000002  # OpenAI API cost per token

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    outputs = Conv2D(6, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

model = get_unet(Input((512, 512, 3)), 64, 0.1)
model.load_weights('InceptionResNetV2-UNet.h5')
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

def get_ai_analysis(current_data, previous_data=None):
    """Get OpenAI analysis of the forest coverage data"""
    try:
        if previous_data:
            prompt = f"""
            Analyze this forest coverage data and provide insights:
            
            Current Report:
            - Forest coverage: {current_data['forested_percentage']:.2f}%
            - Non-forest areas: {current_data['deforested_percentage']:.2f}%
            - Other areas: {current_data['other_percentage']:.2f}%
            
            Previous Report:
            - Forest coverage: {previous_data['forested_percentage']:.2f}%
            - Non-forest areas: {previous_data['deforested_percentage']:.2f}%
            - Other areas: {previous_data['other_percentage']:.2f}%
            
            Please provide:
            1. Key changes and trends
            2. Potential environmental impact
            3. Recommendations for conservation
            """
        else:
            prompt = f"""
            Analyze this forest coverage data and provide insights:
            
            Current Report:
            - Forest coverage: {current_data['forested_percentage']:.2f}%
            - Non-forest areas: {current_data['deforested_percentage']:.2f}%
            - Other areas: {current_data['other_percentage']:.2f}%
            
            Please provide:
            1. Analysis of current forest coverage
            2. Potential environmental implications
            3. Recommendations for conservation
            """

        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}])

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting AI analysis: {str(e)}")
        return "AI analysis currently unavailable."

def calculate_costs(tile_count):
    """Calculate costs for generating the report"""
    prediction_cost = COST_PER_PREDICTION
    tile_costs = tile_count * COST_PER_TILE
    ai_analysis_cost = 0.02  # Approximate cost for OpenAI API call

    total_cost = prediction_cost + tile_costs + ai_analysis_cost
    return {
        "prediction_cost": prediction_cost,
        "tile_costs": tile_costs,
        "ai_analysis_cost": ai_analysis_cost,
        "total_cost": total_cost
    }

def generate_comparison_image(original_image, masked_image):
    # Resize both images to be the same size
    target_size = (512, 512)
    original_resized = cv2.resize(original_image, target_size)
    masked_resized = cv2.resize(masked_image, target_size)

    # Create figure with white background
    plt.figure(figsize=(12, 8), facecolor='white')
    
    # Remove any extra spacing
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    # Create side-by-side subplot for images with white background
    plt.subplot(121)
    plt.imshow(original_resized)
    plt.title('Satellite Image')
    plt.axis('off')
    plt.gca().set_facecolor('white')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(masked_resized, cv2.COLOR_BGR2RGB))
    plt.title('Forest Analysis')
    plt.axis('off')
    plt.gca().set_facecolor('white')

    # Add legend below the images
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#3C1098', label='Building'),
        plt.Rectangle((0,0),1,1, facecolor='#8429F6', label='Land (unpaved)'),
        plt.Rectangle((0,0),1,1, facecolor='#6EC1E4', label='Road'),
        plt.Rectangle((0,0),1,1, facecolor='#FEDD3A', label='Vegetation'),
        plt.Rectangle((0,0),1,1, facecolor='#E2A929', label='Water'),
        plt.Rectangle((0,0),1,1, facecolor='#9B9B9B', label='Unlabeled')
    ]
    
    # Add legend with white background
    plt.figlegend(handles=legend_elements, loc='lower center', ncol=3, 
                 bbox_to_anchor=(0.5, 0.0), facecolor='white')

    # Remove any extra whitespace
    plt.tight_layout()
    
    # Save with white background and no extra space
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
    """Get human-readable location name from coordinates"""
    try:
        # Add a user agent to avoid being blocked
        headers = {
            'User-Agent': 'ReLeaf Forest Monitor/1.0'
        }
        response = requests.get(
            f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}",
            headers=headers
        )
        if response.status_code == 200:
            data = response.json()
            address = data.get('address', {})

            # Try to build the most meaningful location string
            location_parts = []

            # Try different levels of detail
            for key in ['village', 'town', 'city', 'county', 'state_district', 'state', 'country']:
                if address.get(key):
                    location_parts.append(address[key])

            if location_parts:
                return ", ".join(location_parts[:3])  # Return up to 3 levels of detail

    except Exception as e:
        print(f"Error getting location name: {e}")

    return "Unknown Location"  # Fallback if geocoding fails

def create_html_report(region_data, current_results, previous_results, comparison_image, 
                      pie_chart, ai_analysis, costs):
    """Create HTML formatted report"""

    # Format date and time
    current_time = datetime.now()
    formatted_date = current_time.strftime('%B %d, %Y')  # e.g., March 19, 2024
    formatted_time = current_time.strftime('%I:%M %p')   # e.g., 02:30 PM
    if region_data == 'xxx':
        location = "Custom (One time)"
    else:
        # Get location name
        location = get_location_name(
            region_data['min_lat'], 
            region_data['min_long']
        )
    

    # Create comparison section based on whether previous results exist
    if previous_results:
        # Check if values are different
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
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>ReLeaf Forest Coverage Report</h1>
                    <p>{formatted_date} at {formatted_time}</p>
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

                <div class="section">
                    <h2>Report Details</h2>
                    <div class="costs">
                        <h3>Processing Information</h3>
                        <p>Prediction Processing: ${costs['prediction_cost']:.3f}</p>
                        <p>Satellite Imagery: ${costs['tile_costs']:.3f}</p>
                        <p>AI Analysis: ${costs['ai_analysis_cost']:.3f}</p>
                        <p><strong>Total Cost: ${costs['total_cost']:.3f}</strong></p>
                    </div>
                </div>
            </div>
        </body>
    </html>
    """
    return html_template

def send_email_report(email, region_data, prediction_results,map=True,image=""):
    try:
        # Get previous results if they exist
        previous_results = None
        if map:
            if email in watchlist and region_data.get('previous_results'):
                previous_results = region_data['previous_results']
                subject = f"ReLeaf Forest Coverage Report - {region_data['timeframe']}"
            else:
                subject = f"ReLeaf Forest Coverage Report - One Time"
            # Create visualizations
            original_image = fetch_and_stitch_tiles(
                region_data['min_lat'],
                region_data['max_lat'],
                region_data['min_long'],
                region_data['max_long'],
                region_data['zoom']
            )
            tile_count = ((region_data['max_lat'] - region_data['min_lat']) * 
                 (region_data['max_long'] - region_data['min_long']) * 
                 (2 ** region_data['zoom']))
        else:
            subject = "ReLeaf Forest Coverage Report - Custom (One Time)"
            original_image=image
            tile_count = 4
        # Decode masked image
        masked_image = cv2.imdecode(
            np.frombuffer(
                base64.b64decode(prediction_results['predicted_mask_base64']),
                np.uint8
            ),
            cv2.IMREAD_COLOR
        )

        # Generate images
        comparison_image = generate_comparison_image(original_image, masked_image)
        pie_chart_image = generate_pie_chart(
            prediction_results['forested_percentage'],
            prediction_results['deforested_percentage'],
            prediction_results['other_percentage']
        )

        # Get AI analysis and calculate costs
        ai_analysis = get_ai_analysis(prediction_results, previous_results)
        costs = calculate_costs(tile_count)

        # Create HTML report
        html_content = create_html_report(
            region_data,
            prediction_results,
            previous_results,
            "comparison_image.png",
            "pie_chart.png",
            ai_analysis,
            costs
        )

        # Create mail object with attachments
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

        # Send email using Mailtrap client
        client = MailtrapClient(token=MAILTRAP_API_KEY)
        client.send(mail)
        print(f"Successfully sent report to {email}")
        return comparison_image

    except Exception as e:
        print(f"Failed to send email to {email}: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

def create_legend():
    """Create a legend showing what each color represents"""
    legend_height = 150
    legend_width = 200
    legend = Image.new('RGB', (legend_width, legend_height), 'white')
    draw = ImageDraw.Draw(legend)
    
    # Define colors and their labels
    colors = {
        'Building': '#3C1098',
        'Land (unpaved)': '#8429F6',
        'Road': '#6EC1E4',
        'Vegetation': '#FEDD3A',
        'Water': '#E2A929',
        'Unlabeled': '#9B9B9B'
    }
    
    # Draw color boxes and labels
    y_offset = 10
    box_size = 20
    for label, color in colors.items():
        draw.rectangle([10, y_offset, 10 + box_size, y_offset + box_size], fill=color)
        draw.text((40, y_offset + 2), label, fill='black')
        y_offset += box_size + 5
    
    return legend

def process_image(image):
    """Process image and return prediction results with legend"""
    # Preprocess image
    image_preprocessed = preprocess_image(image)
    prediction = model.predict(image_preprocessed)[0]
    
    # Create output image
    output_image = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    
    # Define class mapping
    class_mapping = {
        0: 'building',      # Index 0 = Building
        1: 'land',          # Index 1 = Land
        2: 'road',          # Index 2 = Road
        3: 'vegetation',    # Index 3 = Vegetation
        4: 'water',         # Index 4 = Water
        5: 'unlabeled'      # Index 5 = Unlabeled
    }
    
    # Define colors in BGR format
    colors = {
        'building': (41, 169, 226),    # #c7a23c
        'land': (246, 41, 132),        # #4b3800
        'road': (228, 193, 110),       # #66635b
        'vegetation': (152, 16, 60),   # #0e5400
        'water': (58, 221, 254),       # #005ec9
        'unlabeled': (155, 155, 155)   # #000000
    }
    
    # Get class predictions
    class_indices = np.argmax(prediction, axis=-1)
    
    # Print debug information
    print("Class distribution:")
    for idx in range(6):
        mask = (class_indices == idx)
        percentage = np.sum(mask) / class_indices.size * 100
        print(f"Class {idx} ({class_mapping[idx]}): {percentage:.2f}%")
    
    # Map predictions to colors
    for idx, class_name in class_mapping.items():
        mask = (class_indices == idx)
        output_image[mask] = colors[class_name]
    
    # Calculate percentages for grouped classes
    total_pixels = prediction.shape[0] * prediction.shape[1]
    
    # Forested (Vegetation - class 3)
    forested_mask = (class_indices == 4)
    forested_percentage = np.sum(forested_mask) / total_pixels * 100
    
    # Deforested (Building - class 0, Land - class 1)
    deforested_mask = (class_indices == 0) | (class_indices == 1)
    deforested_percentage = np.sum(deforested_mask) / total_pixels * 100
    
    # Other (Road - class 2, Water - class 4, Unlabeled - class 5)
    other_mask = (class_indices == 2) | (class_indices == 3) | (class_indices == 5)
    other_percentage = np.sum(other_mask) / total_pixels * 100
    
    print(f"\nGrouped percentages:")
    print(f"Forested (Vegetation): {forested_percentage:.2f}%")
    print(f"Deforested (Building + Land): {deforested_percentage:.2f}%")
    print(f"Other (Road + Water + Unlabeled): {other_percentage:.2f}%")
    
    combined_image = output_image
    
    # Convert to base64
    _, buffer = cv2.imencode('.png', combined_image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "predicted_mask_base64": base64_image,
        "forested_percentage": float(forested_percentage),
        "deforested_percentage": float(deforested_percentage),
        "other_percentage": float(other_percentage)
    }

def preprocess_image(image):
    """Preprocess image for model prediction"""
    image_resized = cv2.resize(image, (512, 512))
    image_normalized = image_resized / 255.0
    return np.expand_dims(image_normalized, axis=0)

def lat_lon_to_tile(lat, lon, zoom):
    """Convert latitude/longitude to tile coordinates"""
    n = 2.0 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2.0 * n)
    return x_tile, y_tile

def fetch_and_stitch_tiles(min_lat, max_lat, min_lon, max_lon, zoom):
    """Fetch and stitch together map tiles for the given coordinates"""
    tile_size = 256

    # Convert bounds to tile coordinates
    x_min, y_max = lat_lon_to_tile(max_lat, min_lon, zoom)
    x_max, y_min = lat_lon_to_tile(min_lat, max_lon, zoom)

    # Ensure correct ordering
    y_min, y_max = min(y_min, y_max), max(y_min, y_max)

    # Calculate stitched image dimensions
    stitched_width = (x_max - x_min + 1) * tile_size
    stitched_height = (y_max - y_min + 1) * tile_size

    # Create new image
    stitched_image = Image.new("RGB", (stitched_width, stitched_height))

    # Fetch and stitch tiles
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            url = f"https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={zoom}&s=Ga"
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                tile = Image.open(io.BytesIO(response.content))
                stitched_image.paste(
                    tile, ((x - x_min) * tile_size, (y - y_min) * tile_size)
                )

    # Convert to numpy array
    return np.array(stitched_image)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        email = data['email']
        results = None  # Placeholder for processed results
        
        if 'image' in data:
            # Case 3: User uploads a custom image
            image = base64.b64decode(data['image'].encode('utf-8'))
            image = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            custom = True
        else:
            custom = False
            # Case 1 & 2: User selects a region on the map
            image = fetch_and_stitch_tiles(
                data['min_lat'],
                data['max_lat'],
                data['min_long'],
                data['max_long'],
                data['zoom']
            )

        # Process image through AI model
        results = process_image(image)
        results['ai_analysis'] = get_ai_analysis(results)

        # Case 1: User adds region to watchlist
        if data.get('watchlist'):
            if email not in watchlist:
                watchlist[email] = {}

            region_id = f"region_{len(watchlist[email])}"  # Unique ID for region

            # Store region in the watchlist
            watchlist[email][region_id] = {
                'min_lat': data['min_lat'],
                'max_lat': data['max_lat'],
                'min_long': data['min_long'],
                'max_long': data['max_long'],
                'zoom': data['zoom'],
                'timeframe': data['timeframe'],
                'next_report': get_next_report_time(data['timeframe']).isoformat(),
                'previous_results': {
                    'forested_percentage': results['forested_percentage'],
                    'deforested_percentage': results['deforested_percentage'],
                    'other_percentage': results['other_percentage']
                }
            }

            # Update email report with specific watchlist region
            comparison_image = send_email_report(email, watchlist[email][region_id], results)
            results['predicted_mask_base64'] = base64.b64encode(comparison_image).decode('utf-8')

            save_watchlist()  # Save changes to watchlist
        else:
            if custom:
                # Send email report and retrieve comparison image
                comparison_image = send_email_report(email, 'xxx', results,image=image,map=False)  # Default case (xxx placeholder)
            else:
                # Store region in the watchlist
                region_data = {
                    'min_lat': data['min_lat'],
                    'max_lat': data['max_lat'],
                    'min_long': data['min_long'],
                    'max_long': data['max_long'],
                    'zoom': data['zoom'],
                    'previous_results': ''
                }
                comparison_image = send_email_report(email, region_data, results)
            if comparison_image is None:
                raise ValueError("Comparison image is None.")
    
            results['predicted_mask_base64'] = base64.b64encode(comparison_image).decode('utf-8')
        return jsonify(results)

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

def check_and_send_reports():
    while True:
        current_time = datetime.now()

        for email, regions in list(watchlist.items()):
            for region_id, region_data in list(regions.items()):
                next_report = datetime.fromisoformat(region_data['next_report'])

                if current_time >= next_report:
                    # Get new prediction
                    image = fetch_and_stitch_tiles(
                        region_data['min_lat'],
                        region_data['max_lat'],
                        region_data['min_long'],
                        region_data['max_long'],
                        region_data['zoom']
                    )

                    # Process image and get results
                    results = process_image(image)

                    # Store current results as previous for next time
                    previous_results = region_data.get('previous_results')
                    watchlist[email][region_id]['previous_results'] = {
                        'forested_percentage': results['forested_percentage'],
                        'deforested_percentage': results['deforested_percentage'],
                        'other_percentage': results['other_percentage']
                    }

                    # Send email report
                    send_email_report(email, region_data, results)

                    # Update next report time
                    watchlist[email][region_id]['next_report'] = get_next_report_time(
                        region_data['timeframe']
                    ).isoformat()

                    # Save updated watchlist
                    save_watchlist()

        time.sleep(3600)  # Check every hour

if __name__ == "__main__":
    # Start the reporting thread
    reporting_thread = threading.Thread(target=check_and_send_reports, daemon=True)
    reporting_thread.start()

    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port=8080)
