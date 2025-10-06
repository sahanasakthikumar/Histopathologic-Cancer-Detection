---

## **4. Cloud Deployment**
```markdown
# Web Application & Cloud Deployment

## Overview
Flask-based web interface for cancer detection with comparison between CPU and GPU models. Infrastructure deployed to Azure using Terraform.

## Files
### Web Application (`app/`)
- `app.py` - Flask backend with model loading and prediction
- `templates/index.html` - Interactive web UI for image upload

### Infrastructure as Code (`terraform/`)
- `main.tf` - Complete Azure infrastructure definition
- Virtual machine, networking, storage, and security configuration

## Features
- Upload histopathology images 
- Real-time prediction from both CPU and GPU models
- Side-by-side model comparison
- Inference time measurement
- Confidence scores and accuracy display

## Architecture
- **Cloud Provider**: Microsoft Azure
- **Region**: East Asia
- **VM Size**: Standard_B1s (free tier eligible)
- **OS**: Ubuntu 22.04 LTS
- **Storage**: Azure Blob Storage for model files
