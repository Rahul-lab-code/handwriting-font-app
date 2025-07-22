# Handwriting Font Application

This project aims to create a machine learning model that takes images of handwritten notes and generates a custom font based on the user's handwriting. The application processes handwritten text, extracts characters, and generates a font file that can be used in various applications.

## Project Structure

```
handwriting-font-app
├── data
│   ├── raw_pages              # Original images of handwritten notes
│   ├── segmented_characters    # Segmented character images
│   ├── generated_characters    # Generated character images from the model
│   └── fonts                   # Final font files generated
├── src
│   ├── main.py                 # Entry point of the application
│   ├── char_segmentation.py     # Functions for character extraction
│   ├── labeling.py              # Functions for labeling segmented characters
│   ├── dataset.py               # Handwriting dataset class for training
│   ├── model.py                 # CVAE model definition
│   ├── train.py                 # Model training process
│   ├── generate.py              # Generate new character images
│   ├── svg_convert.py           # Convert images to SVG format
│   └── fontforge_script.py      # Script for generating .ttf font files
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd handwriting-font-app
   ```

2. **Install the required dependencies:**
   Create a virtual environment and install the dependencies listed in `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```

3. **Prepare your handwritten notes:**
   Place your handwritten notes images in the `data/raw_pages` directory.

4. **Run the application:**
   Execute the main script to start the process:
   ```
   python src/main.py
   ```

## Usage Guidelines

- The application will automatically segment characters from the images in the `data/raw_pages` directory and save them in the `data/segmented_characters` directory.
- You will need to manually label the segmented characters, which will be saved in a CSV file.
- The model will be trained using the labeled dataset, and new character images will be generated.
- Finally, the generated character images will be converted to SVG format and compiled into a .ttf font file.

## Overview of Functionality

- **Character Segmentation:** Extracts individual characters from handwritten notes using contour detection.
- **Labeling:** Maps segmented character images to their corresponding labels for training.
- **Dataset Preparation:** Loads and preprocesses character images and labels for the model.
- **Model Training:** Trains a Conditional Variational Autoencoder (CVAE) to generate character variations.
- **Character Generation:** Samples from the trained model to create new character images.
- **SVG Conversion:** Converts generated character images to SVG format for font creation.
- **Font Generation:** Uses FontForge to convert SVG files into a .ttf font file.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.