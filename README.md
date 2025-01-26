# 🌱 **ReLeaf: AI for Environmental Insights** 🌍  
ReLeaf is an AI-driven platform that transforms high-resolution satellite and drone imagery into actionable insights for combating deforestation, fostering reforestation, and analyzing land use changes. Harnessing advanced machine learning, ReLeaf empowers environmental decision-makers to drive impactful conservation efforts.

---

## 📂 **Repository Structure**

- **`client/`**: Frontend application enabling user interaction and data visualization.  
- **`server/`**: Backend service for data processing and AI model inference.  

---

## 🧠 **Model Overview**

ReLeaf employs state-of-the-art deep learning techniques for precise image segmentation and analysis, making complex environmental insights accessible to users. 

### Model Details

- **Architecture**: U-Net-based, fine-tuned for multi-class segmentation.  
- **Input**: High-resolution satellite or drone images (512x512 pixels).  
- **Output Classes**:  
  - **0**: Building `[60, 16, 152]`  
  - **1**: Land `[132, 41, 246]`  
  - **2**: Road `[110, 193, 228]`  
  - **3**: Vegetation `[254, 221, 58]`  
  - **4**: Water `[226, 169, 41]`  
  - **5**: Unlabeled `[155, 155, 155]`  
- **Loss Function**: Combination of Dice Loss and Categorical Cross-Entropy for robust training.  
- **Dataset**:  
  - [Humans in the Loop Semantic Segmentation Dataset](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery/)  
  - Training insights based on [Satellite Segmentation Notebook](https://www.kaggle.com/code/kahtan/satellite-segmentation).  
- **Performance**: Achieves high segmentation accuracy and reliable generalization across test datasets.  

---

## 🌟 **Key Features**

- **Intelligent Imagery Analysis**: Converts raw satellite and drone imagery into actionable land use classifications.  
- **Cutting-Edge Segmentation**: Produces pixel-accurate segmentation masks with clear categorical breakdowns.  
- **User-Centric Design**: Accessible and intuitive interface for non-technical users.  
- **Optimized Workflow**: Robust server-client architecture ensures smooth data processing.  

---

## 🚀 **Setup Instructions**

### Prerequisites

1. **Python 3.8+**  
2. **Dependencies**: Install required libraries:  
   ```bash
   pip install -r requirements.txt
   ```
3. **API Keys**: Obtain the following API keys for full functionality:  
   - Google Geocoding API  
   - OpenAI API  
   - Mailtrap API

### Steps

#### Server Setup

1. Navigate to the `server/` directory:  
   ```bash
   cd server
   ```
2. Run the server application:  
   ```bash
   python main.py
   ```
3. **Note**: The server supports both local (with CUDA-enabled GPU) and cloud-based setups (e.g., GCP). Ensure the necessary API keys are added to the environment.

#### Client Setup

1. Navigate to the `client/` directory:  
   ```bash
   cd client
   ```
2. Run the client application:  
   ```bash
   python main.py
   ```
3. **Configuration**: Update the server's IP address in the `main.py` file and ensure the Google Geocoding API key is correctly configured.  
4. Open the client interface and select satellite or upload drone imagery for analysis.

---

## 🤝 **Contributing**

Contributions are welcomed! To propose improvements or add features:
- Fork this repository.
- Create a new branch.
- Submit a pull request.  

UI design inspiration is attributed to [HotinGo](https://github.com/Just-Moh-it/HotinGo); all implementation details are crafted by the ReLeaf team.

---

## 📜 **License**

ReLeaf is released under the [MIT License](./LICENSE). Feel free to use, modify, and distribute this project with proper attribution.

---

## 📞 **Contact**

For inquiries or support, please open an issue via the GitHub [Issues tab](./issues).
