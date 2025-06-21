
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# --- Generator Model Definition (must match training script) ---
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_channels=1, image_size=28):
        super(Generator, self).__init__()
        
        self.init_size = image_size // 4  # 7 for 28x28
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(True),
            nn.Conv2d(64, num_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z.view(z.size(0), -1))
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# --- Load Model ---
@st.cache_resource
def load_generator():
    device = torch.device("cpu")  # Use CPU for deployment
    generator = Generator(latent_dim=100, num_channels=1, image_size=28)
    
    try:
        # Load the trained model weights
        generator.load_state_dict(torch.load('generator_mnist_dcgan.pth', map_location=device))
        generator.eval()
        return generator, device
    except FileNotFoundError:
        st.error("Model file 'generator_mnist_dcgan.pth' not found. Please ensure the model file is in the same directory as this app.")
        return None, device

# --- Streamlit App ---
def main():
    st.title("🎲 Handwritten Digit Generator")
    st.markdown("*Generate realistic handwritten digits using a trained DCGAN model*")
    
    # Load model
    generator, device = load_generator()
    
    if generator is None:
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("Generation Controls")
    
    # Number of digits to generate
    num_digits = st.sidebar.slider("Number of digits to generate:", 1, 25, 9)
    
    # Random seed for reproducibility
    if st.sidebar.checkbox("Use random seed"):
        seed = st.sidebar.number_input("Seed value:", min_value=0, max_value=99999, value=42)
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate button
    if st.sidebar.button("🎯 Generate New Digits", type="primary"):
        st.session_state.generate_new = True
    
    # Auto-generate on first load
    if 'generate_new' not in st.session_state:
        st.session_state.generate_new = True
    
    if st.session_state.generate_new:
        with st.spinner("Generating digits..."):
            # Generate random noise
            noise = torch.randn(num_digits, 100, device=device)
            
            # Generate images
            with torch.no_grad():
                generated_images = generator(noise).cpu().numpy()
            
            # Denormalize from [-1, 1] to [0, 1]
            generated_images = (generated_images + 1) / 2.0
            
            # Display results
            st.header("Generated Digits")
            
            # Calculate grid dimensions
            cols = min(5, num_digits)
            rows = (num_digits + cols - 1) // cols
            
            # Create matplotlib figure
            fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes]
            elif cols == 1:
                axes = [[ax] for ax in axes]
            
            for i in range(num_digits):
                row = i // cols
                col = i % cols
                
                if rows == 1:
                    ax = axes[col] if cols > 1 else axes[0]
                else:
                    ax = axes[row][col] if cols > 1 else axes[row][0]
                
                ax.imshow(generated_images[i].squeeze(), cmap='gray')
                ax.axis('off')
                ax.set_title(f'Digit {i+1}', fontsize=10)
            
            # Hide unused subplots
            for i in range(num_digits, rows * cols):
                row = i // cols
                col = i % cols
                if rows == 1:
                    ax = axes[col] if cols > 1 else axes[0]
                else:
                    ax = axes[row][col] if cols > 1 else axes[row][0]
                ax.axis('off')
            
            plt.tight_layout()
            
            # Display in Streamlit
            st.pyplot(fig)
            plt.close()
            
            # Display individual images in columns
            st.subheader("Individual Generated Digits")
            columns = st.columns(min(5, num_digits))
            
            for i in range(num_digits):
                col_idx = i % 5
                with columns[col_idx]:
                    # Convert to PIL Image for better display
                    img_array = (generated_images[i].squeeze() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_array, mode='L')
                    # Resize for better visibility
                    pil_img = pil_img.resize((112, 112), Image.NEAREST)
                    st.image(pil_img, caption=f'Generated Digit {i+1}', use_column_width=True)
        
        # Reset the generation flag
        st.session_state.generate_new = False
    
    # Information section
    with st.expander("ℹ️ About this App"):
        st.markdown("""
        This web application uses a **Deep Convolutional Generative Adversarial Network (DCGAN)** 
        trained on the MNIST dataset to generate realistic handwritten digits.
        
        **How it works:**
        1. The model takes random noise as input
        2. A Generator network transforms this noise into realistic digit images
        3. The Generator was trained adversarially against a Discriminator network
        
        **Features:**
        - Generate 1-25 digits at once
        - Optional random seed for reproducible results
        - High-quality 28x28 pixel digit generation
        
        **Model Architecture:**
        - Latent dimension: 100
        - Generator: Linear + Upsampling + Convolutional layers
        - Trained on MNIST dataset (60,000 handwritten digits)
        """)

if __name__ == "__main__":
    main()
