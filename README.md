# Semiocity_Agent — 2D→3D Generation Explorations 

This is part of the Academic work at Iaac Barcelona for the Master's Program im Advance Computation for Architecture and Design 2024-2025

## Team Members: 

### Jose, Francesco, and Paul 

Purpuse :

<img width="1333" height="750" alt="image" src="https://github.com/user-attachments/assets/61bcf820-3063-4ae5-8cca-158fbed983ab" />


Technical Data and Use Imstructions:

A streamlined AI system for generating immersive 3D worlds from 2D images using a unified PyQt6 interface.

🏗️ Architecture Overview
The system uses a unified approach with a single main application that integrates all workflow components:

Core Components
Main Application: RUN_SemioCity_UI.py - Unified PyQt6 interface
3D Viewer: 3d_viewer_simple_working.html - Web-based 3D visualization
Fallback Viewer: 3d_viewer_self_contained_fixed.html - Self-contained 3D viewer
API Keys: api/keys.py - Hugging Face and other API credentials
Workflow Modules
01_ai_generation/ - Edited FLUX image generation
02_remove_background/ - Edited FLUX Kontext background removal
03_2D_to_3D/ - Edited Hunyuan3D 3D generation backend
🚀 Quick Start
Prerequisites and tested on
Conda enviroment Python 3.11.3
Windows 11
CUDA-enabled GPU (recommended)
Hugging Face account and token
Downloaded Three.js repository add to root folder: git clone --depth=1 https://github.com/mrdoob/three.js.git
Installation
Clone the repository:
git clone <repository-url>
cd semiotic_world_agent
Set up API keys: Edit api/keys.py with your Hugging Face token:
HF_TOKEN = "your_hugging_face_token_here"
Install dependencies:
# Install all dependencies (consolidated requirements file)
pip install -r requirements.txt

# Alternative: Manual installation of core packages
pip install PyQt6 PyQt6-WebEngine torch torchvision diffusers transformers
pip install trimesh open3d rembg pillow numpy python-dotenv opencv-python
pip install accelerate safetensors huggingface-hub scikit-image scipy
Note: The requirements.txt file contains all necessary dependencies with specific versions to ensure compatibility. It includes 47 packages covering UI, AI/ML, image processing, 3D generation, and utilities.

Run the application:
python RUN_SemioCity_UI.py
📁 Current Project Structure
semiotic_world_agent/
├── RUN_SemioCity_UI.py # GUI + orchestrator (CLI flags optional) ├── web/ │ └── 3d_viewer.html # single viewer, CDN or offline ├── requirements.txt # pinned deps ├── .env.example # sample env (do NOT commit .env) ├── api/ │ └── keys.py # reads tokens from env ├── shared/ │ ├── init.py │ ├── common.py │ └── settings.py # central defaults (size, steps, models) ├── 01_ai_generation/ │ └── run_flux_workflow.py ├── 02_remove_background/ │ └── flux_kontext_simple.py ├── 03_2D_to_3D/ │ ├── 3d_backend/ │ │ ├── hy3dshape/ │ │ └── hy3dpaint/ │ └── hunyuan3d21/ └── output/ ├── images/ ├── meshes/ # .glb recommended ├── textured/ └── depth/


## **🎯 Key Features**

### **Unified Interface**

- **Single Application**: One PyQt6 app handles the entire pipeline
- **Dark Theme**: Modern, professional UI design
- **Real-time Progress**: Live updates during processing
- **3D Viewer Integration**: Built-in 3D model visualization

### **Workflow Pipeline**

1. **Image Generation** - FLUX-based AI image creation
2. **Background Removal** - FLUX Kontext cleanup
3. **3D Generation** - Hunyuan3D 2D→3D conversion
4. **3D Visualization** - Live 3D model viewing

### **3D Viewer Options**

- **Primary**: `3d_viewer_simple_working.html` (Three.js with CDN)
- **Fallback**: `3d_viewer_self_contained_fixed.html` (self-contained, no CDN)

## **🔧 Configuration**

## **API Keys Setup**

API_Read.md for more information 



### **Model Configuration**
- **Image Generation**: FLUX.1-schnell (fast) or FLUX.1-dev (quality)
- **Background Removal**: FLUX.1-Kontext-dev
- **3D Generation**: Hunyuan3D-2.1

### **Output Settings**
- **Image Size**: 1024x1024 (configurable)
- **3D Format**: GLB (recommended)

### **Dependencies**
The project uses a single consolidated requirements file:
- **`requirements.txt`** - Complete dependencies with specific versions (47 packages)

**Key Dependencies Include:**
- **UI Framework**: PyQt6, PyQt6-WebEngine
- **AI/ML**: PyTorch, Transformers, Diffusers, Hugging Face Hub
- **Image Processing**: Pillow, OpenCV, NumPy, SciPy, Scikit-Image
- **3D Processing**: Trimesh, Open3D, PyMeshLab
- **Background Removal**: RemBG
- **Utilities**: Python-dotenv, TQDM, Requests

**Note**: All individual module requirements files have been consolidated into the main requirements.txt file for simplified dependency management.

## **🎮 Usage Guide**

### **Basic Workflow**

1. **Launch**: Run `python RUN_SemioCity_UI.py`
2. **Enter Prompt**: Type your image description in the AI Prompt field
3. **Adjust Steps**: Set generation steps (1-50, default: 10)
4. **Generate Image**: Click "🎨 Generate Image (FLUX)"
5. **Remove Background**: Click "✂️ Remove Background (Kontext)"
6. **Create 3D**: Click "💎 Generate 3D Mesh"
7. **View 3D**: Use the built-in 3D viewer to interact with your model

### **Advanced Options**

- **Load Existing Image**: Use "📁 Load Image" button
- **Full Pipeline**: Click "🚀 Run Complete 2D→3D Pipeline"
- **Test 3D Viewer**: Use "🔧 Test 3D" button
- **Download Mesh**: Use "⬇️" button to save your 3D model

### **3D Viewer Controls**

- **Mouse**: Rotate, zoom, pan around the 3D model
- **Test Button**: Load most recent mesh for testing
- **Download**: Save mesh file to your computer

## **🛠️ Technical Details**

### **Workflow Integration**

- **Embedded Orchestrator**: All workflows integrated in `RUN_SemioCity_UI.py`
- **Hugging Face Authentication**: Automatic token loading for all models
- **Error Handling**: Comprehensive error recovery and user feedback
- **Progress Tracking**: Real-time status updates

### **3D Generation Backend**

- **Primary**: Edited Hunyuan3D drop-in (`hunyuan3d21.dropin.image_to_3d`)
- **Fallback**: Edited Direct pipelines (`hy3dshape` + `hy3dpaint`)
- **Output Formats**: GLB, OBJ, PLY support
- **Quality Control**: Automatic mesh validation

### **Performance Optimization**

- **GPU Acceleration**: CUDA support for all models
- **Memory Management**: Efficient model loading and cleanup
- **Caching**: Hugging Face model caching
- **Batch Processing**: Support for multiple images

## **🔍 Troubleshooting**

### **Common Issues**

**"3D viewer not loading"**
- Try the "🔧 Test 3D" button
- Check if mesh files exist in `output/meshes/`
- Verify WebEngine permissions are enabled

**"Generation fails"**
- Verify Hugging Face token in `api/keys.py`
- Check GPU memory availability
- Ensure all dependencies are installed

**"WebEngine not available"**
- Install PyQt6-WebEngine: `pip install PyQt6-WebEngine`
- Use Python fallback viewer instead

### **Debug Mode**
- Check status log in the application
- Look for error messages in terminal
- Verify file paths and permissions

## **📊 Performance Tips**

### **GPU Optimization**
- Use CUDA-enabled GPU for best performance
- Monitor GPU memory usage
- Close other GPU-intensive applications

### **Memory Management**
- Process one image at a time for large models
- Clear output folder periodically
- Use appropriate image sizes

### **Speed Optimization**
- Use FLUX.1-schnell for faster image generation
- Adjust generation steps (1-50) for speed vs quality balance
- Enable texture generation only when needed

## **🔄 Recent Updates**

### **v2.1.0 - Enhanced UI & Workflow**
- ✅ **Implementation of alternative workflows for the UI**: Inter font, better styling, and layout
- ✅ **Steps Control**: Configurable generation steps (1-50)
- ✅ **3D Viewer Fixes**: Proper viewport sizing and model framing
- ✅ **WebEngine Permissions**: Fixed local file and CDN access
- ✅ **Clean Structure**: Removed test files and unused components

### **Key Improvements**
- **Better Quality Control**: Adjustable generation steps for quality vs speed
- **Enhanced 3D Viewing**: Fixed viewport issues and model positioning
- **Improved UI**: Modern Inter font and professional styling
- **Better Error Handling**: Comprehensive logging and user feedback


## **📄 License**

This project is licensed under the MIT License - see the LICENSE file for details.


**Ready to create amazing 3D worlds from 2D images!** 🚀✨
