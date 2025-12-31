import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import io
from datetime import datetime
import json
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

st.set_page_config(page_title="SpillGuard AI", page_icon="🌊", layout="wide")

st.markdown("""
    <style>
        .main { background-color: #0e1117; }
        h1 { color: #00e5ff; text-align: center; }
        .stButton>button { width: 100%; background-color: #ff4b4b; color: white; font-weight: bold; }
        .stMetric { background-color: #262730; padding: 10px; border-radius: 5px; }
        .image-info { background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin: 10px 0; }
        .info-text { color: #00e5ff; font-size: 14px; margin: 5px 0; }
    </style>
    """, unsafe_allow_html=True)

IMG_SIZE = (256, 256)
THRESHOLD = 0.6
ALPHA = 0.4
NOISE_SIZE = 3

# Initialize session state for history
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

@st.cache_resource
def load_model_cached():
    try:
        return tf.keras.models.load_model('oil_spill_model.h5', compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    img_resized = cv2.resize(img_array, IMG_SIZE)
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch, img_resized

def create_heatmap(prediction):
    heatmap = np.uint8(255 * prediction)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_PLASMA)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

def plot_histogram(prediction):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(prediction.flatten(), bins=50, color='skyblue', alpha=0.7)
    ax.set_title('Pixel Confidence Distribution')
    ax.set_xlabel('Probability')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    return fig

def plot_confidence_distribution(raw_prediction):
    """Create an easier-to-understand confidence score distribution"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Categorize predictions
    high_conf = np.sum((raw_prediction >= 0.7).astype(int))
    medium_conf = np.sum((raw_prediction >= 0.4) & (raw_prediction < 0.7))
    low_conf = np.sum(raw_prediction < 0.4)
    
    categories = ['Low\n(<0.4)', 'Medium\n(0.4-0.7)', 'High\n(≥0.7)']
    values = [low_conf, medium_conf, high_conf]
    colors_bar = ['#90EE90', '#FFD700', '#FF6B6B']
    
    bars = ax.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Pixels', fontweight='bold')
    ax.set_xlabel('Confidence Level', fontweight='bold')
    ax.set_title('AI Confidence Distribution', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def calculate_metrics(binary_mask, raw_prediction, threshold=0.6):
    """Calculate precision, recall, F1-score, IoU, and Dice coefficient"""
    predicted_positive = (raw_prediction > threshold).astype(int).flatten()
    mask_positive = (binary_mask > 0).astype(int).flatten()
    
    # True Positives, False Positives, False Negatives
    tp = np.sum((predicted_positive == 1) & (mask_positive == 1))
    fp = np.sum((predicted_positive == 1) & (mask_positive == 0))
    fn = np.sum((predicted_positive == 0) & (mask_positive == 1))
    tn = np.sum((predicted_positive == 0) & (mask_positive == 0))
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # IoU (Intersection over Union)
    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union > 0 else 0
    
    # Dice Coefficient
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1_score * 100,
        'iou': iou * 100,
        'dice': dice * 100,
        'accuracy': accuracy * 100
    }

def generate_pdf_report(image, overlay, binary_mask, heatmap_img, metrics, coverage, image_info, timestamp):
    """Generate comprehensive PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#00e5ff'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#ff4b4b'),
        spaceAfter=12,
        fontName='Helvetica-Bold'
    )
    
    # Title
    story.append(Paragraph("🌊 SpillGuard AI Analysis Report", title_style))
    story.append(Spacer(1, 12))
    
    # Timestamp
    story.append(Paragraph(f"<b>Analysis Date:</b> {timestamp}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Image Information
    story.append(Paragraph("📷 Image Information", heading_style))
    img_data = [
        ['Property', 'Value'],
        ['File Name', image_info['name']],
        ['File Size', image_info['size']],
        ['Resolution', image_info['resolution']],
        ['Dimensions', image_info['dimensions']]
    ]
    img_table = Table(img_data, colWidths=[2*inch, 3*inch])
    img_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(img_table)
    story.append(Spacer(1, 20))
    
    # Detection Results
    story.append(Paragraph("🎯 Detection Results", heading_style))
    status = "CRITICAL" if coverage > 5.0 else "Monitor"
    results_data = [
        ['Metric', 'Value'],
        ['Oil Coverage', f"{coverage:.2f}%"],
        ['Alert Level', status],
        ['Risk Assessment', 'High Risk' if status == "CRITICAL" else 'Normal']
    ]
    results_table = Table(results_data, colWidths=[2*inch, 3*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(results_table)
    story.append(Spacer(1, 20))
    
    # Performance Metrics
    story.append(Paragraph("📊 Performance Metrics", heading_style))
    metrics_data = [
        ['Metric', 'Score (%)'],
        ['Precision', f"{metrics['precision']:.2f}%"],
        ['Recall', f"{metrics['recall']:.2f}%"],
        ['F1-Score', f"{metrics['f1_score']:.2f}%"],
        ['IoU (Jaccard)', f"{metrics['iou']:.2f}%"],
        ['Dice Coefficient', f"{metrics['dice']:.2f}%"],
        ['Accuracy', f"{metrics['accuracy']:.2f}%"]
    ]
    metrics_table = Table(metrics_data, colWidths=[2*inch, 3*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    # Add images
    story.append(PageBreak())
    story.append(Paragraph("🖼️ Visual Analysis", heading_style))
    story.append(Spacer(1, 12))
    
    # Save images temporarily
    def pil_to_reportlab(pil_img, width=5*inch):
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        return RLImage(img_buffer, width=width)
    
    story.append(Paragraph("<b>Detection Overlay:</b>", styles['Normal']))
    story.append(pil_to_reportlab(Image.fromarray(overlay)))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("<b>Binary Segmentation Mask:</b>", styles['Normal']))
    story.append(pil_to_reportlab(Image.fromarray(binary_mask)))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("<b>Probability Heatmap:</b>", styles['Normal']))
    story.append(pil_to_reportlab(Image.fromarray(heatmap_img)))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

st.title("🌊 SpillGuard: AI Oil Spill Detection")

model = load_model_cached()

if model is not None:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Get file info
        file_size_bytes = uploaded_file.size
        file_size_mb = file_size_bytes / (1024 * 1024)
        file_size_str = f"{file_size_mb:.2f} MB" if file_size_mb >= 1 else f"{file_size_bytes / 1024:.2f} KB"
        
        image_info = {
            'name': uploaded_file.name,
            'size': file_size_str,
            'resolution': f"{image.size[0]} × {image.size[1]} pixels",
            'dimensions': f"{image.size[0]} × {image.size[1]}"
        }
        
        col1, col2, col3 = st.columns([1, 0.2, 1])
        with col1:
            st.subheader("Original Satellite Image")
            st.image(image, use_container_width=True)

        with col3:
            st.subheader("Action")
            st.write("")
            st.write("")
            analyze_btn = st.button("🔍 Run Analysis", use_container_width=True)
            
            # Display image info below button
            st.markdown("---")
            st.markdown("### 📋 Image Information")
            st.markdown(f"""
            <div class='image-info'>
                <p class='info-text'><b>📄 File Name:</b> {image_info['name']}</p>
                <p class='info-text'><b>💾 File Size:</b> {image_info['size']}</p>
                <p class='info-text'><b>📐 Resolution:</b> {image_info['resolution']}</p>
            </div>
            """, unsafe_allow_html=True)

        if analyze_btn:
            with st.spinner('🔄 Processing image...'):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                processed_input, original_size_img = preprocess_image(image)
                raw_prediction = model.predict(processed_input, verbose=0)[0] 
                
                # Thresholding & Noise Removal
                binary_mask = (raw_prediction > THRESHOLD).astype(np.uint8) * 255
                if NOISE_SIZE > 1:
                    kernel = np.ones((NOISE_SIZE, NOISE_SIZE), np.uint8)
                    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                
                # Visuals Generation
                heatmap_img = create_heatmap(raw_prediction)
                
                mask_colored = np.zeros_like(original_size_img)
                mask_colored[:, :, 0] = 255 
                
                mask_indices = np.squeeze(binary_mask > 0)
                overlay = original_size_img.copy()
                
                if np.any(mask_indices):
                    overlay[mask_indices] = (original_size_img[mask_indices] * (1-ALPHA) + 
                                           mask_colored[mask_indices] * ALPHA).astype(np.uint8)

                oil_px = np.count_nonzero(binary_mask)
                total_px = binary_mask.size
                coverage = (oil_px / total_px) * 100
                
                # Calculate metrics
                metrics = calculate_metrics(binary_mask, raw_prediction, THRESHOLD)
                
                # Save to history
                analysis_record = {
                    'timestamp': timestamp,
                    'image_name': image_info['name'],
                    'coverage': coverage,
                    'metrics': metrics,
                    'status': "CRITICAL" if coverage > 5.0 else "Monitor"
                }
                st.session_state.analysis_history.append(analysis_record)

            st.success("✅ Analysis complete!")
            st.write("---")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis Dashboard", "👁️ Visual Inspection", "📈 Performance Metrics", "📜 History"])
            
            with tab1:
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.image(overlay, caption="AI Detection Overlay (Red = Oil)", width=600)
                
                with col_b:
                    st.metric("Oil Coverage", f"{coverage:.2f}%")
                    status = "CRITICAL" if coverage > 5.0 else "Monitor"
                    st.metric("Alert Level", status, delta="High Risk" if status=="CRITICAL" else "Normal", delta_color="inverse")
                    st.metric("Analysis Time", timestamp)
                    
                    st.write("---")
                    
                    # Generate PDF
                    pdf_buffer = generate_pdf_report(image, overlay, binary_mask, heatmap_img, 
                                                     metrics, coverage, image_info, timestamp)
                    
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"spillguard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    # Download overlay image
                    overlay_pil = Image.fromarray(overlay)
                    buf = io.BytesIO()
                    overlay_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="🖼️ Download Overlay Image",
                        data=byte_im,
                        file_name="spillguard_overlay.png",
                        mime="image/png",
                        use_container_width=True
                    )

            with tab2:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(binary_mask, caption="Binary Segmentation Mask", use_container_width=True, clamp=True)
                with c2:
                    st.image(heatmap_img, caption="Probability Heatmap", use_container_width=True)

            with tab3:
                # Display metrics
                st.subheader("🎯 Model Performance Metrics")
                
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Precision", f"{metrics['precision']:.2f}%")
                    st.metric("Recall", f"{metrics['recall']:.2f}%")
                
                with metric_cols[1]:
                    st.metric("F1-Score", f"{metrics['f1_score']:.2f}%")
                    st.metric("IoU (Jaccard)", f"{metrics['iou']:.2f}%")
                
                with metric_cols[2]:
                    st.metric("Dice Coefficient", f"{metrics['dice']:.2f}%")
                    st.metric("Accuracy", f"{metrics['accuracy']:.2f}%")
                
                st.write("---")
                
                # Charts
                g1, g2 = st.columns(2)
                with g1:
                    st.pyplot(plot_histogram(raw_prediction))
                with g2:
                    st.pyplot(plot_confidence_distribution(raw_prediction))
                
                # Metrics explanation
                with st.expander("ℹ️ Metrics Explanation"):
                    st.markdown("""
                    - **Precision**: Percentage of detected oil pixels that are actually oil
                    - **Recall**: Percentage of actual oil pixels that were detected
                    - **F1-Score**: Harmonic mean of Precision and Recall (overall performance)
                    - **IoU (Intersection over Union)**: Overlap between predicted and actual oil regions
                    - **Dice Coefficient**: Similarity measure between predicted and actual masks
                    - **Accuracy**: Overall correctness of the model's predictions
                    """)
            
            with tab4:
                st.subheader("📜 Analysis History")
                
                if len(st.session_state.analysis_history) > 0:
                    # Display history in reverse chronological order
                    for idx, record in enumerate(reversed(st.session_state.analysis_history)):
                        with st.expander(f"🕐 {record['timestamp']} - {record['image_name']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Coverage:** {record['coverage']:.2f}%")
                                st.write(f"**Status:** {record['status']}")
                            with col2:
                                st.write(f"**F1-Score:** {record['metrics']['f1_score']:.2f}%")
                                st.write(f"**IoU:** {record['metrics']['iou']:.2f}%")
                    
                    # Export history button
                    history_json = json.dumps(st.session_state.analysis_history, indent=2)
                    st.download_button(
                        label="💾 Export History (JSON)",
                        data=history_json,
                        file_name=f"analysis_history_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
                    
                    # Clear history button
                    if st.button("🗑️ Clear History"):
                        st.session_state.analysis_history = []
                        st.rerun()
                else:
                    st.info("No analysis history yet. Run an analysis to see records here.")