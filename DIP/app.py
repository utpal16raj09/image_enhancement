import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import rembg

# Streamlit app
def main():
    st.title('Image Editor')
    st.write('Upload an image to get started!')

    # Upload image
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Sidebar options
        st.sidebar.header('Edit Options')
        option = st.sidebar.selectbox('Choose an option:', ['Background Removal', 'Background Replacement', 'Image Enhancements', 'Artistic Filters'])

        if option == 'Background Removal':
            st.write('Removing background...')
            input_array = np.array(image)
            output_array = rembg.remove(input_array)
            output_image = Image.fromarray(output_array)
            st.image(output_image, caption='Background Removed', use_column_width=True)
            st.download_button('Download Image', output_image.tobytes(), file_name='background_removed.png')

        elif option == 'Background Replacement':
            bg_file = st.file_uploader('Upload background image', type=['jpg', 'png', 'jpeg'])
            if bg_file is not None:
                bg_image = Image.open(bg_file)
                input_array = np.array(image)
                output_array = rembg.remove(input_array)
                output_image = Image.fromarray(output_array)
                output_image = output_image.resize(bg_image.size)
                combined = Image.alpha_composite(bg_image.convert('RGBA'), output_image)
                st.image(combined, caption='Background Replaced', use_column_width=True)
                st.download_button('Download Image', combined.tobytes(), file_name='background_replaced.png')

        elif option == 'Image Enhancements':
            brightness = st.slider('Brightness', 0.5, 1.5, 1.0)
            contrast = st.slider('Contrast', 0.5, 1.5, 1.0)
            sharpness = st.slider('Sharpness', 0.0, 2.0, 1.0)
            blur = st.slider('Blur', 0, 10, 0)

            enhancer = ImageEnhance.Brightness(image)
            enhanced = enhancer.enhance(brightness)
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(contrast)
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(sharpness)
            if blur > 0:
                enhanced = enhanced.filter(ImageFilter.GaussianBlur(blur))

            st.image(enhanced, caption='Enhanced Image', use_column_width=True)
            st.download_button('Download Image', enhanced.tobytes(), file_name='enhanced.png')

        elif option == 'Artistic Filters':
            filter_type = st.selectbox('Choose a filter:', ['Greyscale', 'Cartoon', 'Sketch', 'Neon Glow'])
            img_array = np.array(image)

            if filter_type == 'Greyscale':
                filtered = image.convert('L')
            elif filter_type == 'Cartoon':
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
                color = cv2.bilateralFilter(img_array, 9, 300, 300)
                cartoon = cv2.bitwise_and(color, color, mask=edges)
                filtered = Image.fromarray(cartoon)
            elif filter_type == 'Sketch':
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                inverted = 255 - gray
                blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
                inverted_blurred = 255 - blurred
                sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
                filtered = Image.fromarray(sketch)
            elif filter_type == 'Neon Glow':
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                neon = cv2.addWeighted(img_array, 0.7, edges, 0.3, 0)
                filtered = Image.fromarray(neon)

            st.image(filtered, caption=f'{filter_type} Filter', use_column_width=True)
            st.download_button('Download Image', filtered.tobytes(), file_name=f'{filter_type.lower()}.png')

if __name__ == '__main__':
    main()
